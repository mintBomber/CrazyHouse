"""
Piece definitions, color enum, Move dataclass, and action-space encoding.

Action index layout (total 4416):
  0 .. 4095  : board moves  → from_sq * 64 + to_sq  (from_sq = row*8+col)
  4096 .. 4415 : drop moves → 4096 + piece_idx*64 + to_sq
                  piece_idx is index in DROPPABLE_PIECES
"""
from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple


class PieceType(IntEnum):
    PAWN   = 1
    ROOK   = 2
    KNIGHT = 3
    BISHOP = 4
    QUEEN  = 5
    KING   = 6


class Color(IntEnum):
    WHITE =  1
    BLACK = -1

    def opponent(self) -> "Color":
        return Color.BLACK if self == Color.WHITE else Color.WHITE


# Pieces that can be held in hand and dropped (King is never captured)
DROPPABLE_PIECES: list[PieceType] = [
    PieceType.PAWN,
    PieceType.ROOK,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.QUEEN,
]

ACTION_SPACE_SIZE: int = 64 * 64 + len(DROPPABLE_PIECES) * 64  # 4416

# Short letter for each piece (for display / notation)
PIECE_LETTER: dict[PieceType, str] = {
    PieceType.PAWN:   "P",
    PieceType.ROOK:   "R",
    PieceType.KNIGHT: "N",
    PieceType.BISHOP: "B",
    PieceType.QUEEN:  "Q",
    PieceType.KING:   "K",
}


@dataclass(frozen=True)
class Move:
    """Immutable representation of a single game move."""
    from_pos: Optional[Tuple[int, int]]  # (row, col); None for drop moves
    to_pos:   Tuple[int, int]             # (row, col)
    promotion: Optional[PieceType] = None # Only set for pawn promotion moves
    is_drop:   bool = False
    drop_piece: Optional[PieceType] = None  # Piece being dropped

    def __str__(self) -> str:
        if self.is_drop:
            letter = PIECE_LETTER[self.drop_piece]
            col = chr(ord("a") + self.to_pos[1])
            rank = 8 - self.to_pos[0]
            return f"{letter}@{col}{rank}"
        fc = chr(ord("a") + self.from_pos[1])
        fr = 8 - self.from_pos[0]
        tc = chr(ord("a") + self.to_pos[1])
        tr = 8 - self.to_pos[0]
        promo = f"={PIECE_LETTER[self.promotion]}" if self.promotion else ""
        return f"{fc}{fr}{tc}{tr}{promo}"

    def __repr__(self) -> str:
        return f"Move({self!s})"


# ---------------------------------------------------------------------------
# Action encoding helpers
# ---------------------------------------------------------------------------

def move_to_action_idx(move: Move) -> int:
    to_sq = move.to_pos[0] * 8 + move.to_pos[1]
    if move.is_drop:
        piece_idx = DROPPABLE_PIECES.index(move.drop_piece)
        return 4096 + piece_idx * 64 + to_sq
    from_sq = move.from_pos[0] * 8 + move.from_pos[1]
    return from_sq * 64 + to_sq


def action_idx_to_partial_move(action_idx: int) -> Move:
    """
    Decode an action index to a Move without game-state context.
    For board moves, promotion is NOT set here (caller must set it if needed).
    """
    if action_idx >= 4096:
        drop_offset = action_idx - 4096
        piece_idx   = drop_offset // 64
        to_sq       = drop_offset % 64
        return Move(
            from_pos   = None,
            to_pos     = (to_sq // 8, to_sq % 8),
            is_drop    = True,
            drop_piece = DROPPABLE_PIECES[piece_idx],
        )
    from_sq = action_idx // 64
    to_sq   = action_idx % 64
    return Move(
        from_pos = (from_sq // 8, from_sq % 8),
        to_pos   = (to_sq  // 8, to_sq  % 8),
    )
