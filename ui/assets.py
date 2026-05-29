"""
Asset loading: split chess_pieces.png into individual piece sprites.

chess_pieces.png layout:
  Row 0 (top)    : White pieces — Pawn, Rook, Knight, Bishop, Queen, King
  Row 1 (bottom) : Black pieces — Pawn, Rook, Knight, Bishop, Queen, King

The image is divided into a 2×6 grid.  Because the six pieces are NOT
evenly spaced (the sprite sheet has variable gaps between them), the column
boundaries are detected automatically from the alpha channel rather than
assumed to be sw//6 apart.
"""
from __future__ import annotations

import os
import pygame

from game.pieces import PieceType, Color


# Piece column order in the sprite sheet (left → right)
_PIECE_COLUMNS = [
    PieceType.PAWN,
    PieceType.ROOK,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.QUEEN,
    PieceType.KING,
]

# (color, piece_type) → surface
SpriteDict = dict[tuple[Color, PieceType], pygame.Surface]


def _detect_column_cuts(sheet: pygame.Surface, n_cols: int) -> list[int]:
    """
    Scan the sheet for fully-transparent vertical strips and return a list of
    n_cols+1 x-positions that cleanly separate the pieces.

    The cut between adjacent pieces is placed at the midpoint of the
    transparent gap between them.  Falls back to equal division if the
    expected number of pieces is not found.
    """
    sw, sh = sheet.get_size()
    alpha = pygame.surfarray.array_alpha(sheet)  # shape (sw, sh)

    # Build a boolean mask: True = the entire column is transparent
    transparent = (alpha.max(axis=1) == 0)  # shape (sw,)

    # Locate piece extents as (x_start, x_end) spans
    extents: list[tuple[int, int]] = []
    in_piece = False
    seg_start = 0
    for x in range(sw):
        if not transparent[x] and not in_piece:
            seg_start = x
            in_piece = True
        elif transparent[x] and in_piece:
            extents.append((seg_start, x - 1))
            in_piece = False
    if in_piece:
        extents.append((seg_start, sw - 1))

    if len(extents) != n_cols:
        # Detection failed — fall back to equal-width cells
        return [round(i * sw / n_cols) for i in range(n_cols + 1)]

    # Cut at the midpoint of each inter-piece gap
    cuts = [0]
    for i in range(n_cols - 1):
        mid = (extents[i][1] + extents[i + 1][0]) // 2
        cuts.append(mid)
    cuts.append(sw)
    return cuts


def load_sprites(
    pieces_path: str,
    piece_size: int,
) -> SpriteDict:
    """
    Load and split chess_pieces.png.

    Args:
        pieces_path: absolute path to chess_pieces.png
        piece_size:  desired pixel size (width = height) for each piece sprite

    Returns:
        dict mapping (Color, PieceType) → pygame.Surface
    """
    if not os.path.isfile(pieces_path):
        raise FileNotFoundError(f"Piece image not found: {pieces_path}")

    sheet = pygame.image.load(pieces_path).convert_alpha()
    sw, sh = sheet.get_size()

    x_cuts = _detect_column_cuts(sheet, len(_PIECE_COLUMNS))
    cell_h  = sh // 2

    sprites: SpriteDict = {}

    for row_idx, color in enumerate([Color.WHITE, Color.BLACK]):
        for col_idx, pt in enumerate(_PIECE_COLUMNS):
            x0   = x_cuts[col_idx]
            x1   = x_cuts[col_idx + 1]
            rect = pygame.Rect(x0, row_idx * cell_h, x1 - x0, cell_h)
            cell = sheet.subsurface(rect).copy()
            scaled = pygame.transform.smoothscale(cell, (piece_size, piece_size))
            sprites[(color, pt)] = scaled

    return sprites


def load_board_image(board_path: str, display_width: int) -> pygame.Surface:
    """
    Load the chess board image and scale it to *display_width* pixels wide,
    preserving the original aspect ratio (important: the source image is
    1602×1202, NOT square, so forcing a square scale distorts the grid).
    """
    if not os.path.isfile(board_path):
        raise FileNotFoundError(f"Board image not found: {board_path}")
    img = pygame.image.load(board_path).convert()
    ow, oh = img.get_size()
    display_height = round(oh * display_width / ow)
    return pygame.transform.smoothscale(img, (display_width, display_height))
