"""
Self-play training loop for Gumbel AlphaZero.

Run from the project root:
    python -m ai.train

The loop alternates between:
  1. Self-play : generate game data using Gumbel MCTS.
  2. Training  : optimise the network on the collected (s, π, z) tuples.
  3. Checkpoint: save network weights periodically.

Training target:
  Loss = cross_entropy(π_improved, π_network) + MSE(z, v_network)
  where π_improved is the improved policy from MCTS visit counts and
  z is the actual game outcome (±1 or 0 for draw).
"""
from __future__ import annotations

import os
import sys
import random
import time
import glob
import argparse
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board  import GameState
from game.pieces import Color, ACTION_SPACE_SIZE, move_to_action_idx
from ai.network  import AlphaZeroNet, build_network, state_to_tensor
from ai.mcts     import GumbelMCTS


# Type alias: one training sample
Sample = Tuple[np.ndarray, np.ndarray, float]   # (state_tensor, policy, value)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def self_play_game(
    network: AlphaZeroNet,
    cfg: dict,
    device: str,
) -> List[Sample]:
    """
    Play one game against itself using Gumbel MCTS.
    Returns a list of (state_tensor_CHW, improved_policy, outcome) tuples.
    Outcome is from the perspective of the player who was to move at that state.
    """
    planner   = GumbelMCTS(network, cfg["ai"])
    state     = GameState()
    samples: List[Tuple[np.ndarray, np.ndarray, Color]] = []
    max_moves = cfg["game"]["max_moves"]
    thresh    = cfg["ai"].get("temperature_threshold", 30)

    for move_num in range(max_moves):
        if state.is_terminal():
            break

        temperature = 1.0 if move_num < thresh else 0.0
        probs, _ = planner.run(state, add_noise=True, temperature=temperature)

        # Save (CHW float32 array, improved policy, current_player)
        obs = state_to_tensor(state, device).squeeze(0).cpu().numpy()
        samples.append((obs, probs.copy(), state.current_player))

        # Sample an action proportional to the improved policy
        legal      = state.get_legal_moves()
        legal_idxs = [move_to_action_idx(m) for m in legal]
        lp         = np.array([probs[i] for i in legal_idxs], dtype=np.float64)
        if lp.sum() < 1e-8:
            lp = np.ones(len(legal_idxs)) / len(legal_idxs)
        else:
            lp /= lp.sum()

        chosen = int(np.random.choice(len(legal), p=lp))
        state.apply_move(legal[chosen])

    # Determine outcome
    winner = state.get_winner()   # Color or None

    # Assign z from each stored player's perspective
    result: List[Sample] = []
    for obs, policy, player in samples:
        if winner is None:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0
        result.append((obs, policy, z))

    return result


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    network: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    batch: List[Sample],
    device: str,
) -> Tuple[float, float]:
    """One gradient update. Returns (policy_loss, value_loss)."""
    obs_list    = [s[0] for s in batch]
    policy_list = [s[1] for s in batch]
    value_list  = [s[2] for s in batch]

    obs_t    = torch.tensor(np.stack(obs_list),    dtype=torch.float32).to(device)
    target_p = torch.tensor(np.stack(policy_list), dtype=torch.float32).to(device)
    target_v = torch.tensor(value_list,            dtype=torch.float32).to(device).unsqueeze(1)

    network.train()
    policy_logits, value = network(obs_t)

    # Policy loss: cross-entropy against improved policy (soft target)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_p * log_probs).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value, target_v)

    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def ckpt_dir(cfg: dict) -> str:
    train_cfg = cfg["training"]
    run_name = train_cfg.get("run_name")
    if not run_name:
        raise ValueError("training.run_name is required")
    root = _resolve_project_path(train_cfg.get("checkpoint_dir", "checkpoints"))
    return os.path.join(root, run_name)


def ckpt_path(cfg: dict, iteration: int) -> str:
    return os.path.join(ckpt_dir(cfg), f"iter_{iteration:05d}.pt")


def latest_path(cfg: dict) -> str:
    return os.path.join(ckpt_dir(cfg), "latest.pt")


def _torch_load_checkpoint(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_or_init(cfg: dict, device: str) -> Tuple[AlphaZeroNet, torch.optim.Optimizer, int]:
    train_cfg = cfg["training"]
    network = build_network(cfg["network"], device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    start_iter = 1
    if train_cfg.get("resume", True):
        latest = latest_path(cfg)
        if os.path.isfile(latest):
            ckpt = _torch_load_checkpoint(latest, device)
            network.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            completed_iter = int(ckpt.get("iteration", 0))
            start_iter = completed_iter + 1
            print(f"[train] Resumed from {latest} (iter={completed_iter})")
        else:
            print("[train] No checkpoint found; starting from scratch.")
    else:
        print("[train] resume=false; starting from scratch.")

    return network, optimizer, start_iter


def save_checkpoint(
    cfg: dict,
    network: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> None:
    os.makedirs(ckpt_dir(cfg), exist_ok=True)
    data = {
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "run_name": cfg["training"]["run_name"],
    }
    path = ckpt_path(cfg, iteration)
    torch.save(data, path)
    torch.save(data, latest_path(cfg))
    print(f"  [iter {iteration}] Checkpoint saved: {path}")

    keep = int(cfg["training"].get("keep_last_n_checkpoints", 0))
    if keep > 0:
        _prune_old_checkpoints(cfg, keep)


def _prune_old_checkpoints(cfg: dict, keep: int) -> None:
    files = sorted(glob.glob(os.path.join(ckpt_dir(cfg), "iter_*.pt")))
    for old in files[:-keep]:
        os.remove(old)
        print(f"  [train] Removed old checkpoint: {old}")


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run self-play training using config/config.yaml"
    )
    parser.parse_args(argv)

    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device      = cfg["ai"].get("device", "cpu")
    train_cfg   = cfg["training"]
    cfg["_project_root"] = PROJECT_ROOT
    total_iterations = int(train_cfg.get("total_iterations", 0))
    save_every = int(train_cfg.get("save_every_n_iters", 0))
    num_self_play_games = int(train_cfg["num_self_play_games"])

    if total_iterations < 1:
        raise ValueError("training.total_iterations must be >= 1")
    if save_every < 1:
        raise ValueError("training.save_every_n_iters must be >= 1")
    if num_self_play_games < 1:
        raise ValueError("training.num_self_play_games must be >= 1")

    network, optimizer, start_iter = load_or_init(cfg, device)
    total_steps = total_iterations * num_self_play_games
    completed_steps = (start_iter - 1) * num_self_play_games
    print(f"[train] run={train_cfg['run_name']} total_iterations={total_iterations}")
    print(f"[train] self-play steps={completed_steps} / {total_steps} steps")
    print(f"[train] checkpoint_dir={ckpt_dir(cfg)}")

    buffer: deque = deque(maxlen=train_cfg["buffer_size"])

    if start_iter > total_iterations:
        print(f"[train] Run '{train_cfg['run_name']}' already completed "
              f"through iter={start_iter - 1}.")
        return

    for iteration in range(start_iter, total_iterations + 1):
        t0 = time.time()
        # Self-play phase
        network.eval()
        for game_idx in range(num_self_play_games):
            samples = self_play_game(network, cfg, device)
            buffer.extend(samples)
            completed_steps += 1
            if completed_steps % 10 == 0 or completed_steps == total_steps:
                print(f"{completed_steps} / {total_steps} steps")

        if len(buffer) < train_cfg["batch_size"]:
            print(f"  [iter {iteration}] Buffer too small, skipping training.")
            continue

        # Training phase
        network.train()
        pl_total, vl_total, steps = 0.0, 0.0, 0
        for epoch in range(train_cfg["num_epochs"]):
            batch = random.sample(buffer, min(train_cfg["batch_size"], len(buffer)))
            pl, vl = train_step(network, optimizer, batch, device)
            pl_total += pl; vl_total += vl; steps += 1

        elapsed = time.time() - t0
        print(f"[iter {iteration}]  "
              f"policy_loss={pl_total/steps:.4f}  "
              f"value_loss={vl_total/steps:.4f}  "
              f"time={elapsed:.1f}s")

        if save_every > 0 and (iteration % save_every == 0 or iteration == total_iterations):
            save_checkpoint(cfg, network, optimizer, iteration)

    print(f"[train] Completed {total_iterations} iterations for run "
          f"'{train_cfg['run_name']}'.")


if __name__ == "__main__":
    main()
