"""
AIAgent: wraps a network + GumbelMCTS to make move decisions.

The agent loads/saves model checkpoints and exposes a simple
`select_move(state) -> Move` API used by the UI and training loop.
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch

from game.board  import GameState
from game.pieces import Move, ACTION_SPACE_SIZE, move_to_action_idx
from ai.network  import AlphaZeroNet, build_network
from ai.mcts     import GumbelMCTS


class AIAgent:
    """
    Gumbel AlphaZero agent.

    If no checkpoint exists the network starts with random weights.
    `move_number` is used to decide whether to sample (temperature=1) or
    play greedily (temperature=0) after `temperature_threshold` moves.
    """

    def __init__(self, cfg: dict, checkpoint_path: str | None = None):
        device     = cfg["ai"].get("device", "cpu")
        net_cfg    = cfg["network"]
        self.cfg   = cfg
        self.device = device

        self.network = build_network(net_cfg, device)
        checkpoint_path = _resolve_checkpoint_path(cfg, checkpoint_path)
        self.trained_iteration = 0

        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = _torch_load_checkpoint(checkpoint_path, device)
            self.network.load_state_dict(ckpt["model_state_dict"])
            self.trained_iteration = int(ckpt.get("iteration", 0))
            print(f"[AIAgent] Loaded checkpoint: {checkpoint_path}")
        elif checkpoint_path:
            print(f"[AIAgent] Checkpoint not found: {checkpoint_path}")
            print("[AIAgent] Using random-initialised network.")
        else:
            print("[AIAgent] No checkpoint; using random-initialised network.")

        self.network.eval()
        self.mcts = GumbelMCTS(self.network, cfg["ai"])

    # ------------------------------------------------------------------
    def select_move(self, state: GameState, move_number: int = 999) -> Move:
        """
        Choose a move for the current player using Gumbel MCTS.
        Returns a legal Move object.
        """
        thresh = self.cfg["ai"].get("temperature_threshold", 30)
        temperature = 1.0 if move_number < thresh else 0.0

        probs, _ = self.mcts.run(state, add_noise=False, temperature=temperature)

        legal = state.get_legal_moves()
        if not legal:
            raise ValueError("select_move called with no legal moves")

        # Pick action by sampling from improved policy
        legal_idxs = [move_to_action_idx(m) for m in legal]
        legal_probs = np.array([probs[i] for i in legal_idxs], dtype=np.float64)

        if legal_probs.sum() < 1e-8:
            # Fallback: uniform random
            return random.choice(legal)

        legal_probs /= legal_probs.sum()
        chosen_idx = int(np.random.choice(len(legal), p=legal_probs))
        return legal[chosen_idx]

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str, iteration: int, optimizer_state=None) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "model_state_dict": self.network.state_dict(),
            "iteration": iteration,
        }
        if optimizer_state:
            data["optimizer_state_dict"] = optimizer_state
        torch.save(data, path)
        print(f"[AIAgent] Checkpoint saved: {path}")


def _resolve_checkpoint_path(cfg: dict, checkpoint_path: str | None) -> str | None:
    if checkpoint_path == "":
        return None
    if checkpoint_path is not None:
        return _resolve_project_path(cfg, checkpoint_path)

    train_cfg = cfg.get("training", {})
    ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    run_name = train_cfg.get("run_name", "run_001")
    return _resolve_project_path(cfg, os.path.join(ckpt_dir, run_name, "latest.pt"))


def _resolve_project_path(cfg: dict, path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    base = cfg.get("_project_root") or os.getcwd()
    return os.path.normpath(os.path.join(base, path))


def _torch_load_checkpoint(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
