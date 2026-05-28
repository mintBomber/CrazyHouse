"""
Asset loading: split chess_pieces.png into individual piece sprites.

chess_pieces.png layout:
  Row 0 (top)    : White pieces — Pawn, Rook, Knight, Bishop, Queen, King
  Row 1 (bottom) : Black pieces — Pawn, Rook, Knight, Bishop, Queen, King

The image is divided into a 2×6 grid and each cell is extracted as a
pygame.Surface scaled to the requested size.
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

    cell_w = sw // 6
    cell_h = sh // 2

    sprites: SpriteDict = {}

    for row_idx, color in enumerate([Color.WHITE, Color.BLACK]):
        for col_idx, pt in enumerate(_PIECE_COLUMNS):
            rect = pygame.Rect(col_idx * cell_w, row_idx * cell_h, cell_w, cell_h)
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
