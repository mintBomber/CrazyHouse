"""
Neural network for Gumbel AlphaZero: ResNet with policy and value heads.

Input representation (28 planes, 8×8):
  ch  0– 5 : current player's piece planes (P R N B Q K)
  ch  6–11 : opponent's piece planes       (P R N B Q K)
  ch 12–16 : current player's hand counts  (P R N B Q), value = count/max_count
  ch 17–21 : opponent's hand counts        (P R N B Q)
  ch 22–25 : castling rights               (wK wQ bK bQ), 0 or 1
  ch 26    : en-passant plane              (1 on EP target square)
  ch 27    : side-to-move                  (all 1 if White, all 0 if Black)

The board is always presented from the current player's perspective:
  if Black to move the board is flipped so the current player's pieces are
  always on rows 6–7 and the opponent's on rows 0–1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from game.board  import GameState
from game.pieces import Color, PieceType, DROPPABLE_PIECES, ACTION_SPACE_SIZE

# Max pieces each type can appear in hand
_HAND_MAX = {
    PieceType.PAWN:   8,
    PieceType.ROOK:   2,
    PieceType.KNIGHT: 2,
    PieceType.BISHOP: 2,
    PieceType.QUEEN:  1,
}

INPUT_PLANES   = 28
BOARD_SIZE     = 8


# ---------------------------------------------------------------------------
# State → tensor
# ---------------------------------------------------------------------------

def state_to_tensor(state: GameState, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Convert a GameState to a float32 tensor of shape (1, 28, 8, 8).
    The board is oriented so the current player's pieces are at the bottom.
    """
    player = state.current_player
    opp    = player.opponent()

    board = state.board.copy()
    # Flip board for Black so that current player always "looks up"
    if player == Color.BLACK:
        board = board[::-1, :].copy() * -1   # flip rows, invert signs

    planes = np.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    sign = int(player)
    # Current player pieces
    for i, pt in enumerate(PieceType):
        planes[i]   = (board ==  int(pt)).astype(np.float32)   # ch 0-5
        planes[i+6] = (board == -int(pt)).astype(np.float32)   # ch 6-11

    # Hand counts (normalized)
    for i, pt in enumerate(DROPPABLE_PIECES):
        mx = _HAND_MAX[pt]
        planes[12+i] = state.hands[player][pt] / mx
        planes[17+i] = state.hands[opp][pt]    / mx

    # Castling rights
    for i, right in enumerate(state.castling_rights):   # wK wQ bK bQ
        # Flip castle rights perspective if black to move
        if player == Color.BLACK:
            # Swap white/black castling rights
            mapped = [2, 3, 0, 1][i]
        else:
            mapped = i
        planes[22+i] = float(state.castling_rights[mapped])

    # En passant
    if state.en_passant is not None:
        er, ec = state.en_passant
        if player == Color.BLACK:
            er = 7 - er  # flip row
        planes[26, er, ec] = 1.0

    # Side to move
    planes[27] = 1.0 if player == Color.WHITE else 0.0

    tensor = torch.from_numpy(planes).unsqueeze(0)
    return tensor.to(device)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class AlphaZeroNet(nn.Module):
    """
    ResNet backbone with a policy head and a value head.

    Policy head output: logits over ACTION_SPACE_SIZE actions
    Value head output : scalar in (-1, 1) representing win probability
    """

    def __init__(
        self,
        num_res_blocks: int = 10,
        channels: int = 256,
        policy_channels: int = 2,
        value_channels: int = 1,
        value_fc_size: int = 256,
    ):
        super().__init__()
        self.input_conv = ConvBnRelu(INPUT_PLANES, channels, kernel=3)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = ConvBnRelu(channels, policy_channels, kernel=1)
        self.policy_fc   = nn.Linear(policy_channels * BOARD_SIZE * BOARD_SIZE, ACTION_SPACE_SIZE)

        # Value head
        self.value_conv  = ConvBnRelu(channels, value_channels, kernel=1)
        self.value_fc1   = nn.Linear(value_channels * BOARD_SIZE * BOARD_SIZE, value_fc_size)
        self.value_fc2   = nn.Linear(value_fc_size, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: float tensor of shape (B, INPUT_PLANES, 8, 8)
        Returns:
            policy_logits: (B, ACTION_SPACE_SIZE)
            value:         (B, 1)  ∈ (-1, 1)
        """
        h = self.input_conv(x)
        h = self.res_blocks(h)

        # Policy
        p = self.policy_conv(h)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = self.value_conv(h)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_network(cfg: dict, device: str = "cpu") -> AlphaZeroNet:
    net = AlphaZeroNet(
        num_res_blocks  = cfg.get("num_res_blocks",  10),
        channels        = cfg.get("channels",        256),
        policy_channels = cfg.get("policy_channels",   2),
        value_channels  = cfg.get("value_channels",    1),
        value_fc_size   = cfg.get("value_fc_size",   256),
    )
    return net.to(device)
