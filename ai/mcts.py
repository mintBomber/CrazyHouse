"""
Gumbel AlphaZero MCTS with Sequential Halving.

Reference: "Policy improvement by planning with Gumbel"
           Danihelka et al., ICLR 2022.

Key ideas implemented:
  1. At the root: sample K actions via Gumbel noise + log(prior), then narrow
     down to the best action using Sequential Halving.
  2. At non-root nodes: standard PUCT selection.
  3. Leaf evaluation: network policy + value; backpropagate value with sign flip
     at every ply (alternating perspectives).
  4. The "improved policy" (visit-count distribution) is returned for training.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from game.board  import GameState
from game.pieces import (
    Color, Move, ACTION_SPACE_SIZE, DROPPABLE_PIECES,
    move_to_action_idx, action_idx_to_partial_move,
)
from ai.network import AlphaZeroNet, state_to_tensor


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "is_expanded")

    def __init__(self, prior: float = 0.0):
        self.prior: float  = prior
        self.visit_count: int   = 0
        self.value_sum:   float = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(
        self,
        legal_moves: List[Move],
        policy: np.ndarray,          # shape (ACTION_SPACE_SIZE,)
    ) -> None:
        for mv in legal_moves:
            idx = move_to_action_idx(mv)
            self.children[idx] = MCTSNode(prior=float(policy[idx]))
        self.is_expanded = True

    def best_child_puct(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        """Select child with highest PUCT score."""
        sqrt_n = math.sqrt(self.visit_count + 1)
        best_score = -1e9
        best_idx   = -1
        best_child: Optional[MCTSNode] = None
        for idx, child in self.children.items():
            u = child.q_value + c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            if u > best_score:
                best_score = u
                best_idx   = idx
                best_child = child
        return best_idx, best_child


# ---------------------------------------------------------------------------
# Single MCTS simulation
# ---------------------------------------------------------------------------

def _simulate(
    root_node: MCTSNode,
    root_state: GameState,
    forced_first_action: Optional[int],   # root → force this action first
    network: AlphaZeroNet,
    c_puct: float,
    device: str,
) -> None:
    """Run one tree simulation; update visit counts and values in-place."""
    node  = root_node
    state = root_state.copy()
    path: List[MCTSNode] = [node]

    # --- selection ---
    first = True
    while node.is_expanded and not state.is_terminal():
        if first and forced_first_action is not None:
            action_idx = forced_first_action
            child = node.children.get(action_idx)
            if child is None:
                break  # forced action not legal (shouldn't happen)
            first = False
        else:
            action_idx, child = node.best_child_puct(c_puct)
        path.append(child)
        # Reconstruct the Move from its action index and apply it
        mv = _action_to_move(action_idx, state)
        if mv is None:
            break
        state.apply_move(mv)
        node = child

    # --- expansion + evaluation ---
    if state.is_terminal():
        v = state.get_result()   # from current (about-to-move) perspective
    else:
        legal = state.get_legal_moves()
        if not legal:
            v = state.get_result()
        else:
            obs = state_to_tensor(state, device)
            with torch.no_grad():
                logits, val_t = network(obs)
            v = float(val_t[0, 0].item())

            policy_np = F.softmax(logits[0], dim=0).cpu().numpy()
            # Mask + renormalise to legal moves
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            for m in legal:
                mask[move_to_action_idx(m)] = 1.0
            policy_np = policy_np * mask
            s = policy_np.sum()
            if s > 0:
                policy_np /= s
            else:
                policy_np = mask / mask.sum()

            node.expand(legal, policy_np)

    # --- backpropagation (sign alternates each ply) ---
    for n in reversed(path):
        n.visit_count += 1
        n.value_sum   += v
        v = -v


def _action_to_move(action_idx: int, state: GameState) -> Optional[Move]:
    """Decode an action index to a Move in the context of the current state."""
    mv = action_idx_to_partial_move(action_idx)
    if mv.is_drop:
        return mv
    # Board move: check for pawn promotion
    fr, fc = mv.from_pos
    tr, tc = mv.to_pos
    if 0 <= fr < 8 and 0 <= fc < 8:
        p = int(state.board[fr, fc])
        if abs(p) == 1:  # PAWN
            promo_row = 0 if state.current_player == Color.WHITE else 7
            if tr == promo_row:
                from game.pieces import PieceType
                return Move(mv.from_pos, mv.to_pos, promotion=PieceType.QUEEN)
    return mv


# ---------------------------------------------------------------------------
# Gumbel AlphaZero root search
# ---------------------------------------------------------------------------

class GumbelMCTS:
    """
    Gumbel AlphaZero planner.

    Usage:
        planner = GumbelMCTS(network, cfg['ai'])
        probs, root_value = planner.run(state)
        action_idx = np.argmax(probs)
    """

    def __init__(self, network: AlphaZeroNet, cfg: dict):
        self.network        = network
        self.n_simulations  = cfg.get("mcts_simulations",  400)
        self.K              = cfg.get("gumbel_K",           16)
        self.c_puct         = cfg.get("c_puct",           1.25)
        self.dir_alpha      = cfg.get("dirichlet_alpha",   0.3)
        self.dir_eps        = cfg.get("dirichlet_eps",    0.25)
        self.device         = cfg.get("device",           "cpu")

    def run(
        self,
        state: GameState,
        add_noise: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Run Gumbel Sequential Halving from *state*.

        Returns:
            probs      : improved policy distribution (shape ACTION_SPACE_SIZE,)
            root_value : value estimate for the current player
        """
        legal = state.get_legal_moves()
        if not legal:
            return np.zeros(ACTION_SPACE_SIZE), 0.0

        # --- evaluate root ---
        obs = state_to_tensor(state, self.device)
        with torch.no_grad():
            logits, val_t = self.network(obs)
        root_value = float(val_t[0, 0].item())

        policy_np = F.softmax(logits[0], dim=0).cpu().numpy()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for m in legal:
            mask[move_to_action_idx(m)] = 1.0
        policy_np = policy_np * mask
        s = policy_np.sum()
        policy_np = policy_np / s if s > 0 else mask / mask.sum()

        # Dirichlet noise at root for exploration during self-play
        if add_noise and len(legal) > 1:
            noise = np.random.dirichlet([self.dir_alpha] * len(legal))
            for i, m in enumerate(legal):
                a = move_to_action_idx(m)
                policy_np[a] = (1 - self.dir_eps) * policy_np[a] + self.dir_eps * noise[i]
            s = policy_np.sum()
            policy_np /= s

        # --- expand root ---
        root_node = MCTSNode(prior=1.0)
        root_node.expand(legal, policy_np)

        # --- Gumbel noise: select top-K candidates ---
        K  = min(self.K, len(legal))
        ga = np.random.gumbel(0, 1, len(legal))  # one per legal action
        legal_idxs  = [move_to_action_idx(m) for m in legal]
        legal_prior = np.array([policy_np[a] for a in legal_idxs])
        scores      = ga + np.log(legal_prior + 1e-8)
        top_k_pos   = np.argpartition(scores, -K)[-K:]
        candidates  = [legal[i] for i in top_k_pos]

        # --- Sequential Halving ---
        n_phases = max(1, int(np.floor(np.log2(K))))
        # Budget per phase (we spread n_simulations evenly across phases)
        sims_per_phase_base = max(1, self.n_simulations // n_phases)

        for phase in range(n_phases):
            if len(candidates) <= 1:
                break
            sims_each = max(1, sims_per_phase_base // len(candidates))
            for cand_mv in candidates:
                cand_idx = move_to_action_idx(cand_mv)
                for _ in range(sims_each):
                    _simulate(
                        root_node, state, cand_idx,
                        self.network, self.c_puct, self.device,
                    )
            # Keep top half by Q value
            half = max(1, len(candidates) // 2)
            q_vals = [
                root_node.children[move_to_action_idx(m)].q_value
                for m in candidates
            ]
            keep_idx   = np.argpartition(q_vals, -half)[-half:]
            candidates = [candidates[i] for i in keep_idx]

        # Use any remaining budget on final candidates
        leftover = self.n_simulations - n_phases * sims_per_phase_base
        if leftover > 0 and candidates:
            sims_each = max(1, leftover // len(candidates))
            for cand_mv in candidates:
                cand_idx = move_to_action_idx(cand_mv)
                for _ in range(sims_each):
                    _simulate(
                        root_node, state, cand_idx,
                        self.network, self.c_puct, self.device,
                    )

        # --- build improved policy from visit counts ---
        visits = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for m in legal:
            idx = move_to_action_idx(m)
            child = root_node.children.get(idx)
            if child:
                visits[idx] = float(child.visit_count)

        if temperature == 0:
            best = int(np.argmax(visits))
            probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            probs[best] = 1.0
        else:
            v_temp = visits ** (1.0 / temperature)
            t_sum  = v_temp.sum()
            probs  = v_temp / t_sum if t_sum > 0 else visits / (visits.sum() + 1e-8)

        return probs, root_value
