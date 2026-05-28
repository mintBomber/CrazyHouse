"""
Pygame UI for the chess-shogi hybrid.

Supports three game modes (selected on the menu screen):
  - Player vs Player
  - Player vs AI       (human = White)
  - AI vs AI

Layout:
  ┌────────────────────────────────────────────────────┐
  │  info panel (left)   │   board (center)   │ hand panel (right) │
  └────────────────────────────────────────────────────┘

Controls (human turn):
  - Left-click a piece on the board → select it; legal destination squares are highlighted.
  - Left-click a highlighted destination → execute the move.
  - Left-click a piece in the hand panel → select it for dropping.
  - Left-click a highlighted empty board square → drop the piece.
  - Right-click or click elsewhere → deselect.
  - Promotion: when a pawn reaches the back rank a choice dialog appears.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

import pygame

from game.board  import GameState
from game.pieces import (
    Color, PieceType, Move, DROPPABLE_PIECES,
    move_to_action_idx,
)
from ui.assets import load_sprites, load_board_image


# ---------------------------------------------------------------------------
# Colour constants (overridden by config later)
# ---------------------------------------------------------------------------

_C = {
    "bg":          (40,  40,  40),
    "panel":       (55,  55,  60),
    "text":        (230, 230, 230),
    "text2":       (160, 160, 160),
    "hl_sel":      (80,  200,  80),
    "hl_legal":    (80,  150, 220),
    "hl_last":     (220, 200,  60),
    "hl_check":    (220,  50,  50),
    "hl_hand":     (200, 130,  30),
    "btn":         (70,  100, 140),
    "btn_hover":   (90,  130, 180),
    "btn_txt":     (230, 230, 230),
}


class ChessUI:
    """Main UI class.  Call run() to enter the event loop."""

    # -----------------------------------------------------------------
    def __init__(self, cfg: dict, asset_dir: str, default_ckpt: str | None = None):
        self.cfg       = cfg
        self.asset_dir = asset_dir
        self.default_ckpt = default_ckpt

        ui = cfg["ui"]
        self.W  = ui["window_width"]
        self.H  = ui["window_height"]
        self.BX = ui["board_offset_x"]      # board image left edge in window
        self.BY = ui["board_offset_y"]      # board image top edge in window
        self.BW = ui["board_display_width"] # board image rendered width
        # Aspect ratio: original 1602×1202
        self.BH = round(self.BW * 1202 / 1602)

        # Scale exact pixel positions from the reference 890x668 measurement
        sx = self.BW / 890
        sy = self.BH / 668

        # Pixel x of each column's left edge (within the board image)
        self.COL_PX: List[int] = [int(v * sx) for v in ui["board_col_starts"]]
        # Pixel y of each row's top edge (within the board image)
        self.ROW_PX: List[int] = [int(v * sy) for v in ui["board_row_starts"]]
        # Right / bottom frame boundaries (used for click detection)
        self.FRAME_RIGHT  = int(ui["board_frame_right"]  * sx)
        self.FRAME_BOTTOM = int(ui["board_frame_bottom"] * sy)

        # Width/height of each individual square (varies by ±1 px due to scaling)
        self.COL_SZ: List[int] = [
            (self.COL_PX[c+1] - self.COL_PX[c]) if c < 7
            else (self.FRAME_RIGHT - self.COL_PX[7])
            for c in range(8)
        ]
        self.ROW_SZ: List[int] = [
            (self.ROW_PX[r+1] - self.ROW_PX[r]) if r < 7
            else (self.FRAME_BOTTOM - self.ROW_PX[7])
            for r in range(8)
        ]
        # Piece sprite size — use the minimum square dimension so sprites fit
        self.SQ = ui["board_sq_size"]  # nominal 64 (sprites pre-scaled to this)

        self.FPS = ui["fps"]
        self.HAND_X   = ui["hand_panel_x"]
        self.HAND_SQ  = ui["hand_piece_size"]
        self.ANIM_MS  = ui["animation_speed"]

        # Apply colour overrides from config
        colors = ui.get("colors", {})
        _C["bg"]       = tuple(colors.get("background",         _C["bg"]))
        _C["panel"]    = tuple(colors.get("panel_bg",           _C["panel"]))
        _C["text"]     = tuple(colors.get("text",               _C["text"]))
        _C["text2"]    = tuple(colors.get("text_secondary",     _C["text2"]))
        _C["hl_sel"]   = tuple(colors.get("highlight_selected", _C["hl_sel"]))
        _C["hl_legal"] = tuple(colors.get("highlight_legal",    _C["hl_legal"]))
        _C["hl_last"]  = tuple(colors.get("highlight_last_move",_C["hl_last"]))
        _C["hl_check"] = tuple(colors.get("highlight_check",    _C["hl_check"]))
        _C["hl_hand"]  = tuple(colors.get("hand_selected",      _C["hl_hand"]))
        _C["btn"]      = tuple(colors.get("button_normal",      _C["btn"]))
        _C["btn_hover"]= tuple(colors.get("button_hover",       _C["btn_hover"]))
        _C["btn_txt"]  = tuple(colors.get("button_text",        _C["btn_txt"]))
        self.HL_ALPHA  = ui.get("highlight_alpha", 160)

        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Chess × Shogi  –  Gumbel AlphaZero")
        self.clock  = pygame.time.Clock()

        # Fonts
        fs = ui.get("font_size", 22)
        self.font_md = pygame.font.SysFont("segoeui", fs)
        self.font_sm = pygame.font.SysFont("segoeui", int(fs * 0.8))
        self.font_lg = pygame.font.SysFont("segoeui", int(fs * 1.6), bold=True)

        # Load assets
        self.sprites   = load_sprites(
            os.path.join(asset_dir, "chess_pieces.png"), self.SQ)
        self.board_img = load_board_image(
            os.path.join(asset_dir, "chess_board.png"), self.BW)
        # Smaller sprites for the hand panel
        self.sprites_sm = load_sprites(
            os.path.join(asset_dir, "chess_pieces.png"), self.HAND_SQ)

        # Game state
        self.state: Optional[GameState] = None
        self.game_mode: Optional[str]   = None  # "pvp", "pvai", "aivai"
        self.ai_white: Optional[object] = None  # AIAgent or None
        self.ai_black: Optional[object] = None
        self._shared_agent: Optional[object] = None
        self.game_record: List[dict] = []
        self._game_started_at: Optional[str] = None
        self._last_saved_record: Optional[str] = None
        self._save_btn_rect = pygame.Rect(8, self.H - 150, max(80, self.BX - 20), 34)

        # UI interaction state
        self.selected_square: Optional[Tuple[int,int]] = None  # (row, col)
        self.selected_hand: Optional[Tuple[Color, PieceType]] = None
        self.legal_cache: List[Move] = []
        self.highlight_squares: List[Tuple[int,int]] = []
        self.last_move: Optional[Move] = None

        # AI threading
        self._ai_queue: queue.Queue  = queue.Queue()
        self._ai_busy:  bool         = False

        # Promotion dialog
        self._promo_pending: Optional[Move] = None   # base move awaiting promo choice

        # Move number
        self.move_number = 0

        # Status message
        self.status_msg = ""

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------

    def run(self) -> None:
        self._show_menu()

    # -----------------------------------------------------------------
    # Menu screen
    # -----------------------------------------------------------------

    def _show_menu(self) -> None:
        buttons = [
            ("Player  vs  Player",  "pvp"),
            ("Player  vs  AI",      "pvai"),
            ("AI  vs  AI",          "aivai"),
        ]
        bw, bh = 340, 60
        spacing = 24
        total_h = len(buttons) * (bh + spacing) - spacing
        start_y = (self.H - total_h) // 2 + 60

        running = True
        while running:
            self.screen.fill(_C["bg"])
            # Title
            title = self.font_lg.render("Chess × Shogi", True, _C["text"])
            sub   = self.font_md.render(
                "Gumbel AlphaZero  ·  Piece-drop variant", True, _C["text2"])
            self.screen.blit(title, title.get_rect(center=(self.W//2, self.H//4)))
            self.screen.blit(sub,   sub.get_rect(center=(self.W//2, self.H//4 + 52)))

            mx, my = pygame.mouse.get_pos()
            btn_rects = []
            for i, (label, mode) in enumerate(buttons):
                bx = (self.W - bw) // 2
                by = start_y + i * (bh + spacing)
                rect = pygame.Rect(bx, by, bw, bh)
                btn_rects.append((rect, mode))
                hover = rect.collidepoint(mx, my)
                pygame.draw.rect(self.screen, _C["btn_hover"] if hover else _C["btn"],
                                 rect, border_radius=10)
                txt = self.font_md.render(label, True, _C["btn_txt"])
                self.screen.blit(txt, txt.get_rect(center=rect.center))

            pygame.display.flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for rect, mode in btn_rects:
                        if rect.collidepoint(ev.pos):
                            self._start_game(mode)
                            return

    # -----------------------------------------------------------------
    # Game startup
    # -----------------------------------------------------------------

    def _start_game(self, mode: str) -> None:
        self.game_mode   = mode
        self.state       = GameState()
        self.last_move   = None
        self.move_number = 0
        self.selected_square = None
        self.selected_hand   = None
        self.highlight_squares = []
        self.legal_cache = []
        self.status_msg  = ""
        self._promo_pending = None
        self._ai_busy    = False
        self.game_record = []
        self._game_started_at = datetime.now().isoformat(timespec="seconds")
        self._last_saved_record = None

        if mode == "pvp":
            self.ai_white = None
            self.ai_black = None
        elif mode == "pvai":
            self.ai_white = None                           # human plays White
            self.ai_black = self._get_shared_agent()
        elif mode == "aivai":
            agent = self._get_shared_agent()
            self.ai_white = agent
            self.ai_black = agent

        self._game_loop()

    def _get_shared_agent(self):
        if self._shared_agent is None:
            from ai.agent import AIAgent
            self._shared_agent = AIAgent(self.cfg, self.default_ckpt)
        return self._shared_agent

    # -----------------------------------------------------------------
    # Game loop
    # -----------------------------------------------------------------

    def _game_loop(self) -> None:
        while True:
            self.clock.tick(self.FPS)
            self._handle_events()
            self._check_ai_result()
            self._maybe_start_ai()
            self._draw_frame()
            pygame.display.flip()

            if self.state.is_terminal():
                self._show_game_over()
                return

    # -----------------------------------------------------------------
    # Event handling
    # -----------------------------------------------------------------

    def _handle_events(self) -> None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self._show_menu()
                    return

            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if self._save_btn_rect.collidepoint(ev.pos):
                        self._save_game_record()
                        continue
                    self._on_click(ev.pos)
                elif ev.button == 3:
                    self._deselect()

    def _is_human_turn(self) -> bool:
        p = self.state.current_player
        if p == Color.WHITE:
            return self.ai_white is None
        return self.ai_black is None

    # -----------------------------------------------------------------
    # Click handling
    # -----------------------------------------------------------------

    def _on_click(self, pos: Tuple[int, int]) -> None:
        if not self._is_human_turn() or self._ai_busy:
            return
        if self._promo_pending is not None:
            return  # wait for promotion dialog to be handled separately

        # Check hand panel clicks first
        hand_result = self._hand_click(pos)
        if hand_result is not None:
            color, pt = hand_result
            if color == self.state.current_player:
                if (self.selected_hand == (color, pt)):
                    self._deselect()
                else:
                    self.selected_hand   = (color, pt)
                    self.selected_square = None
                    self._update_highlights()
            return

        # Check board click
        sq = self._pixel_to_square(pos)
        if sq is None:
            self._deselect()
            return

        row, col = sq

        # If a drop is selected, try to drop
        if self.selected_hand is not None:
            _, pt = self.selected_hand
            move = Move(None, (row, col), is_drop=True, drop_piece=pt)
            if (row, col) in self.highlight_squares:
                self._execute_move(move)
            else:
                self._deselect()
            return

        piece = int(self.state.board[row, col])

        if self.selected_square is not None:
            from_sq = self.selected_square
            # Try to move
            if (row, col) in self.highlight_squares:
                self._try_board_move(from_sq, (row, col))
            elif piece != 0 and (piece > 0) == (int(self.state.current_player) > 0):
                # Re-select own piece
                self.selected_square = (row, col)
                self._update_highlights()
            else:
                self._deselect()
        else:
            if piece != 0 and (piece > 0) == (int(self.state.current_player) > 0):
                self.selected_square = (row, col)
                self._update_highlights()
            else:
                self._deselect()

    def _try_board_move(self, from_pos: Tuple[int,int], to_pos: Tuple[int,int]) -> None:
        """Attempt a board move; show promotion dialog if needed."""
        player = self.state.current_player
        fr, fc = from_pos
        piece  = int(self.state.board[fr, fc])
        pt     = PieceType(abs(piece))

        promo_row = 0 if player == Color.WHITE else 7

        if pt == PieceType.PAWN and to_pos[0] == promo_row:
            # Need promotion choice – show dialog
            base_move = Move(from_pos, to_pos)
            self._promo_pending = base_move
            self._draw_frame()
            pygame.display.flip()
            chosen_pt = self._promotion_dialog()
            self._promo_pending = None
            if chosen_pt is None:
                self._deselect()
                return
            move = Move(from_pos, to_pos, promotion=chosen_pt)
        else:
            move = Move(from_pos, to_pos)

        self._execute_move(move)

    def _execute_move(self, move: Move) -> None:
        player = self.state.current_player
        self.game_record.append(self._record_move(move, player, self.move_number + 1))
        self.state.apply_move(move)
        self.last_move   = move
        self.move_number += 1
        self._deselect()

    def _record_move(self, move: Move, player: Color, move_number: int) -> dict:
        return {
            "move_number": move_number,
            "player": "white" if player == Color.WHITE else "black",
            "notation": str(move),
            "from": list(move.from_pos) if move.from_pos is not None else None,
            "to": list(move.to_pos),
            "promotion": move.promotion.name if move.promotion else None,
            "is_drop": move.is_drop,
            "drop_piece": move.drop_piece.name if move.drop_piece else None,
        }

    def _deselect(self) -> None:
        self.selected_square    = None
        self.selected_hand      = None
        self.highlight_squares  = []

    def _update_highlights(self) -> None:
        self.highlight_squares = []
        player = self.state.current_player
        legal  = self.state.get_legal_moves()

        if self.selected_hand is not None:
            _, pt = self.selected_hand
            for m in legal:
                if m.is_drop and m.drop_piece == pt:
                    self.highlight_squares.append(m.to_pos)
        elif self.selected_square is not None:
            for m in legal:
                if not m.is_drop and m.from_pos == self.selected_square:
                    self.highlight_squares.append(m.to_pos)

    # -----------------------------------------------------------------
    # AI management
    # -----------------------------------------------------------------

    def _maybe_start_ai(self) -> None:
        if self._ai_busy or self.state.is_terminal():
            return
        if self._is_human_turn():
            return

        agent = (self.ai_white if self.state.current_player == Color.WHITE
                 else self.ai_black)
        if agent is None:
            return

        self._ai_busy = True
        mn = self.move_number

        def worker():
            try:
                move = agent.select_move(self.state.copy(), mn)
                self._ai_queue.put(("move", move))
            except Exception as e:
                self._ai_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _check_ai_result(self) -> None:
        try:
            tag, payload = self._ai_queue.get_nowait()
        except queue.Empty:
            return

        self._ai_busy = False
        if tag == "move":
            move: Move = payload
            # Small pause so AI moves are visible in AI vs AI
            if self.game_mode == "aivai":
                time.sleep(self.ANIM_MS / 1000.0)
            self._execute_move(move)
        else:
            self.status_msg = f"AI error: {payload}"

    def _save_game_record(self) -> Optional[str]:
        if self.state is None:
            return None

        record_dir = os.path.join(self.asset_dir, "gamerecord")
        os.makedirs(record_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = self.game_mode or "unknown"
        path = os.path.join(record_dir, f"{timestamp}_{mode}_{self.move_number:03d}moves.json")

        winner = self.state.get_winner()
        if winner == Color.WHITE:
            result = "white_win"
        elif winner == Color.BLACK:
            result = "black_win"
        elif self.state.is_terminal():
            result = "draw"
        else:
            result = "in_progress"

        data = {
            "game": "Chess x Shogi",
            "mode": mode,
            "started_at": self._game_started_at,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "move_count": self.move_number,
            "result": result,
            "moves": self.game_record,
            "final_board": repr(self.state),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._last_saved_record = path
        self.status_msg = f"Saved: {os.path.basename(path)}"
        return path

    # -----------------------------------------------------------------
    # Promotion dialog
    # -----------------------------------------------------------------

    def _promotion_dialog(self) -> Optional[PieceType]:
        """Blocking modal dialog for pawn promotion. Returns chosen PieceType."""
        options = [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]
        player  = self.state.current_player

        bw, bh  = self.HAND_SQ + 10, self.HAND_SQ + 10
        spacing = 12
        total_w = len(options) * (bw + spacing) - spacing
        ox      = (self.W - total_w) // 2
        oy      = (self.H - bh) // 2

        while True:
            self._draw_frame()
            # Dim overlay
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            self.screen.blit(overlay, (0, 0))

            prompt = self.font_md.render("Choose promotion piece:", True, _C["text"])
            self.screen.blit(prompt, prompt.get_rect(center=(self.W//2, oy - 36)))

            mx, my  = pygame.mouse.get_pos()
            btn_rects = []
            for i, pt in enumerate(options):
                bx   = ox + i * (bw + spacing)
                rect = pygame.Rect(bx, oy, bw, bh)
                btn_rects.append((rect, pt))
                hover = rect.collidepoint(mx, my)
                pygame.draw.rect(self.screen,
                                 _C["btn_hover"] if hover else _C["btn"],
                                 rect, border_radius=8)
                spr = self.sprites_sm.get((player, pt))
                if spr:
                    self.screen.blit(spr, (bx + 5, oy + 5))

            pygame.display.flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for rect, pt in btn_rects:
                        if rect.collidepoint(ev.pos):
                            return pt

    # -----------------------------------------------------------------
    # Pixel ↔ board coordinate helpers
    # -----------------------------------------------------------------

    def _pixel_to_square(self, pos: Tuple[int,int]) -> Optional[Tuple[int,int]]:
        """Convert a window pixel position to (row, col), or None if outside the grid."""
        x, y = pos
        bx = x - self.BX
        by = y - self.BY
        if not (self.COL_PX[0] <= bx < self.FRAME_RIGHT
                and self.ROW_PX[0] <= by < self.FRAME_BOTTOM):
            return None
        # Binary search for column
        col = 7
        for c in range(7):
            if bx < self.COL_PX[c + 1]:
                col = c
                break
        # Binary search for row
        row = 7
        for r in range(7):
            if by < self.ROW_PX[r + 1]:
                row = r
                break
        return (row, col)

    def _square_to_pixel(self, row: int, col: int) -> Tuple[int,int]:
        """Return the top-left pixel of a board square in window coordinates."""
        return (self.BX + self.COL_PX[col],
                self.BY + self.ROW_PX[row])

    # -----------------------------------------------------------------
    # Hand panel interaction
    # -----------------------------------------------------------------

    def _hand_click(self, pos: Tuple[int,int]) -> Optional[Tuple[Color, PieceType]]:
        """Return (Color, PieceType) if a hand piece was clicked, else None."""
        for color in (Color.WHITE, Color.BLACK):
            rects = self._hand_rects(color)
            for (rect, pt) in rects:
                if rect.collidepoint(pos):
                    return (color, pt)
        return None

    def _hand_rects(self, color: Color) -> List[Tuple[pygame.Rect, PieceType]]:
        """Positions of hand piece slots for *color* in the side panel."""
        sq      = self.HAND_SQ
        padding = 10
        x       = self.HAND_X + padding
        panel_h = self.BH   # hand panel spans the full board image height

        # Black hand at top half, White hand at bottom half of the panel
        if color == Color.BLACK:
            y_start = self.BY + 24
        else:
            y_start = self.BY + panel_h // 2 + 24

        rects: List[Tuple[pygame.Rect, PieceType]] = []
        row, col_offset = 0, 0
        for pt in DROPPABLE_PIECES:
            bx = x + col_offset * (sq + padding)
            by = y_start + row * (sq + padding + 18)
            rects.append((pygame.Rect(bx, by, sq, sq), pt))
            col_offset += 1
            if col_offset >= 2:   # 2 columns to fit inside the narrow side panel
                col_offset = 0
                row += 1
        return rects

    # -----------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------

    def _draw_frame(self) -> None:
        self.screen.fill(_C["bg"])
        self._draw_info_panel()
        self._draw_board()
        self._draw_pieces()
        self._draw_highlights()
        self._draw_hand_panel(Color.WHITE)
        self._draw_hand_panel(Color.BLACK)

    def _draw_board(self) -> None:
        self.screen.blit(self.board_img, (self.BX, self.BY))

    def _draw_pieces(self) -> None:
        for r in range(8):
            for c in range(8):
                val = int(self.state.board[r, c])
                if val == 0:
                    continue
                color = Color.WHITE if val > 0 else Color.BLACK
                pt    = PieceType(abs(val))
                spr   = self.sprites.get((color, pt))
                if spr:
                    px, py = self._square_to_pixel(r, c)
                    # Center the sprite within the actual square dimensions
                    ox = (self.COL_SZ[c] - self.SQ) // 2
                    oy = (self.ROW_SZ[r] - self.SQ) // 2
                    self.screen.blit(spr, (px + ox, py + oy))

    def _hl_surf(self, r: int, c: int) -> pygame.Surface:
        """Create a per-square-sized highlight surface."""
        return pygame.Surface((self.COL_SZ[c], self.ROW_SZ[r]), pygame.SRCALPHA)

    def _draw_highlights(self) -> None:
        # Last move highlight
        if self.last_move:
            for sq in ([self.last_move.from_pos] if self.last_move.from_pos else []) + [self.last_move.to_pos]:
                r, c = sq
                hl = self._hl_surf(r, c)
                hl.fill((*_C["hl_last"], self.HL_ALPHA))
                self.screen.blit(hl, self._square_to_pixel(r, c))

        # Check highlight
        if self.state.is_in_check(self.state.current_player):
            king_val = int(PieceType.KING) * int(self.state.current_player)
            import numpy as np
            pos = np.argwhere(self.state.board == king_val)
            if len(pos):
                kr, kc = int(pos[0, 0]), int(pos[0, 1])
                hl = self._hl_surf(kr, kc)
                hl.fill((*_C["hl_check"], self.HL_ALPHA))
                self.screen.blit(hl, self._square_to_pixel(kr, kc))

        # Selected piece highlight
        if self.selected_square:
            r, c = self.selected_square
            hl = self._hl_surf(r, c)
            hl.fill((*_C["hl_sel"], self.HL_ALPHA))
            self.screen.blit(hl, self._square_to_pixel(r, c))

        # Legal move dots / capture rings
        for (r, c) in self.highlight_squares:
            px, py   = self._square_to_pixel(r, c)
            sw, sh   = self.COL_SZ[c], self.ROW_SZ[r]
            surf     = pygame.Surface((sw, sh), pygame.SRCALPHA)
            surf.fill((0, 0, 0, 0))
            if self.state.board[r, c] != 0:
                # Capture target: draw a ring
                pygame.draw.circle(surf, (*_C["hl_legal"], 160),
                                   (sw // 2, sh // 2), min(sw, sh) // 2 - 2, 4)
            else:
                # Empty target: draw a filled dot
                pygame.draw.circle(surf, (*_C["hl_legal"], 200),
                                   (sw // 2, sh // 2), min(sw, sh) // 5)
            self.screen.blit(surf, (px, py))

    def _draw_hand_panel(self, color: Color) -> None:
        # Background — draw only once for the whole panel (call for BLACK first)
        if color == Color.BLACK:
            panel_rect = pygame.Rect(self.HAND_X, self.BY,
                                     self.W - self.HAND_X - 4, self.BH)
            pygame.draw.rect(self.screen, _C["panel"], panel_rect, border_radius=8)

        # Section title
        label = "White's hand" if color == Color.WHITE else "Black's hand"
        lbl_s = self.font_sm.render(label, True, _C["text2"])
        if color == Color.BLACK:
            self.screen.blit(lbl_s, (self.HAND_X + 10, self.BY + 6))
        else:
            self.screen.blit(lbl_s, (self.HAND_X + 10, self.BY + self.BH // 2 + 6))

        # Piece slots
        hand = self.state.hands[color]
        sq   = self.HAND_SQ
        pad  = 10

        rects = self._hand_rects(color)
        for (rect, pt) in rects:
            # Check if this piece is selected for dropping
            is_sel = (self.selected_hand == (color, pt))
            bg_col = _C["hl_hand"] if is_sel else (70, 70, 75)
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=6)

            spr = self.sprites_sm.get((color, pt))
            count = hand.get(pt, 0)
            # Draw piece sprite (greyed out if count == 0)
            if spr:
                if count == 0:
                    grey = spr.copy()
                    grey.fill((100, 100, 100, 120), special_flags=pygame.BLEND_RGBA_MULT)
                    self.screen.blit(grey, rect.topleft)
                else:
                    self.screen.blit(spr, rect.topleft)

            # Count badge
            count_s = self.font_sm.render(f"×{count}", True,
                                          _C["text"] if count > 0 else _C["text2"])
            self.screen.blit(count_s, (rect.left, rect.bottom + 1))

    def _draw_info_panel(self) -> None:
        # Left panel background
        panel = pygame.Rect(0, 0, self.BX - 4, self.H)
        pygame.draw.rect(self.screen, _C["panel"], panel)

        y = 30
        # Turn indicator
        turn_txt = ("White to move" if self.state.current_player == Color.WHITE
                    else "Black to move")
        turn_col = (230, 230, 230) if self.state.current_player == Color.WHITE else (140, 140, 160)
        t = self.font_md.render(turn_txt, True, turn_col)
        self.screen.blit(t, (10, y)); y += 36

        # AI thinking indicator
        if self._ai_busy:
            thinking = self.font_sm.render("AI thinking…", True, (160, 200, 120))
            self.screen.blit(thinking, (10, y)); y += 28

        y += 10
        # Move number
        mn = self.font_sm.render(f"Move: {self.move_number}", True, _C["text2"])
        self.screen.blit(mn, (10, y)); y += 26

        # Halfmove clock
        hc = self.font_sm.render(f"50-move: {self.state.halfmove_clock}/100",
                                  True, _C["text2"])
        self.screen.blit(hc, (10, y)); y += 26

        # Check indicator
        if self.state.is_in_check(self.state.current_player):
            chk = self.font_md.render("CHECK!", True, _C["hl_check"])
            self.screen.blit(chk, (10, y)); y += 32

        # Save button
        mx, my = pygame.mouse.get_pos()
        hover = self._save_btn_rect.collidepoint(mx, my)
        pygame.draw.rect(self.screen, _C["btn_hover"] if hover else _C["btn"],
                         self._save_btn_rect, border_radius=6)
        save_s = self.font_sm.render("Save GameRecord", True, _C["btn_txt"])
        self.screen.blit(save_s, save_s.get_rect(center=self._save_btn_rect.center))

        # Game mode
        y = self.H - 80
        mode_labels = {"pvp": "Player vs Player", "pvai": "Player vs AI",
                       "aivai": "AI vs AI"}
        mode_s = self.font_sm.render(mode_labels.get(self.game_mode, ""),
                                     True, _C["text2"])
        self.screen.blit(mode_s, (10, y)); y += 24

        esc_s = self.font_sm.render("[ESC] menu", True, _C["text2"])
        self.screen.blit(esc_s, (10, y))

        # Status message
        if self.status_msg:
            sm = self.font_sm.render(self.status_msg, True, (220, 120, 60))
            self.screen.blit(sm, (10, self.H - 110))

    # -----------------------------------------------------------------
    # Game-over screen
    # -----------------------------------------------------------------

    def _show_game_over(self) -> None:
        winner = self.state.get_winner()
        if winner == Color.WHITE:
            headline = "White wins!"
            col = (230, 220, 180)
        elif winner == Color.BLACK:
            headline = "Black wins!"
            col = (140, 140, 160)
        else:
            if self.state.is_stalemate():
                headline = "Stalemate — Draw"
            else:
                headline = "Draw"
            col = _C["text"]

        bw, bh = 260, 56

        while True:
            self._draw_frame()

            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            ht = self.font_lg.render(headline, True, col)
            self.screen.blit(ht, ht.get_rect(center=(self.W//2, self.H//2 - 60)))

            mx, my  = pygame.mouse.get_pos()
            buttons = [
                (pygame.Rect((self.W - bw)//2, self.H//2,       bw, bh), "Save GameRecord"),
                (pygame.Rect((self.W - bw)//2, self.H//2 + 70,  bw, bh), "Play again"),
                (pygame.Rect((self.W - bw)//2, self.H//2 + 140, bw, bh), "Main menu"),
            ]
            for rect, label in buttons:
                hover = rect.collidepoint(mx, my)
                pygame.draw.rect(self.screen, _C["btn_hover"] if hover else _C["btn"],
                                 rect, border_radius=10)
                lbl = self.font_md.render(label, True, _C["btn_txt"])
                self.screen.blit(lbl, lbl.get_rect(center=rect.center))

            if self.status_msg:
                sm = self.font_sm.render(self.status_msg, True, (230, 210, 150))
                self.screen.blit(sm, sm.get_rect(center=(self.W//2, self.H//2 + 220)))

            pygame.display.flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for rect, label in buttons:
                        if rect.collidepoint(ev.pos):
                            if label == "Save GameRecord":
                                self._save_game_record()
                            elif label == "Play again":
                                self._start_game(self.game_mode)
                                return
                            else:
                                self._show_menu()
                                return
