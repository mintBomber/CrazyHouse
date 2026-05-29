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
from typing import Any, List, Optional, Tuple

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
        # Logical canvas: fixed design resolution — all drawing uses these coords.
        # window_width/height in config is only the initial OS window size;
        # _flip() scales the logical canvas to whatever the window actually is.
        self.W = 1200
        self.H = 750
        _init_w = ui.get("window_width",  self.W)
        _init_h = ui.get("window_height", self.H)
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
        pygame.display.set_mode((_init_w, _init_h), pygame.RESIZABLE)
        self.screen = pygame.Surface((self.W, self.H))
        pygame.display.set_caption("Crazy House")
        self.clock  = pygame.time.Clock()

        # Fonts
        fs = ui.get("font_size", 22)
        self.font_md = pygame.font.SysFont("segoeui", fs, bold=True)
        self.font_sm = pygame.font.SysFont("segoeui", int(fs * 0.8), bold=True)
        self.font_xs = pygame.font.SysFont("segoeui", int(fs * 0.65), bold=True)
        self.font_lg = pygame.font.SysFont("segoeui", int(fs * 1.6), bold=True)

        # Load assets
        self.sprites   = load_sprites(
            os.path.join(asset_dir, "chess_pieces.png"), self.SQ)
        self.sprites = self._apply_piece_outlines(self.sprites)
        self.board_img = load_board_image(
            os.path.join(asset_dir, "chess_board.png"), self.BW)
        # Smaller sprites for the hand panel
        self.sprites_sm = load_sprites(
            os.path.join(asset_dir, "chess_pieces.png"), self.HAND_SQ)
        self.sprites_sm = self._apply_piece_outlines(self.sprites_sm, radius=1)

        # Game state
        self.state: Optional[GameState] = None
        self.game_mode: Optional[str]   = None  # "pvp", "pvai", "aivai"
        self.ai_white: Optional[object] = None  # AIAgent or None
        self.ai_black: Optional[object] = None
        self._shared_agent: Optional[object] = None
        self.game_record: List[dict] = []
        self._game_started_at: Optional[str] = None
        self._last_saved_record: Optional[str] = None
        self._resign_btn_rect = pygame.Rect(8, self.H - 190, max(80, self.BX - 20), 34)
        self._save_btn_rect = pygame.Rect(8, self.H - 150, max(80, self.BX - 20), 34)
        self._front_winrate: Optional[float] = None
        self._ai_candidates: Optional[Tuple] = None
        self._value_eval_dirty = True
        self._time_winner: Optional[Color] = None
        self._resign_winner: Optional[Color] = None
        self._clock_remaining = {
            Color.WHITE: 5 * 60.0,
            Color.BLACK: 5 * 60.0,
        }
        self._turn_started_at = time.monotonic()

        # Pre-game settings
        self._setup_drop_mode: bool = True          # True=Crazy House, False=Standard
        self._setup_first_color = Color.WHITE
        self._setup_human_side = "first"
        self._setup_main_minutes = 5
        self._setup_byoyomi_seconds = 10
        self._setup_active_field: Optional[str] = None
        self._setup_input_text = ""
        self.first_color = Color.WHITE
        self.human_color = Color.WHITE
        self.player1_color = Color.WHITE
        self.main_time_sec = 5 * 60.0
        self.byoyomi_sec = 10.0

        # Replay state
        self._is_replay = False
        self._replay_moves: List[dict] = []
        self._replay_index = 0
        self._replay_auto = False
        self._replay_last_step = 0.0
        self._replay_source_name = ""

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

    def _apply_piece_outlines(
        self,
        sprites: dict[tuple[Color, PieceType], pygame.Surface],
        radius: int = 2,
    ) -> dict[tuple[Color, PieceType], pygame.Surface]:
        outlined = {}
        for (color, pt), surf in sprites.items():
            outline_color = (0, 0, 0) if color == Color.WHITE else (255, 255, 255)
            outlined[(color, pt)] = self._outline_surface(surf, outline_color, radius)
        return outlined

    def _outline_surface(
        self,
        surf: pygame.Surface,
        color: tuple[int, int, int],
        radius: int,
    ) -> pygame.Surface:
        base = surf.convert_alpha()
        mask = pygame.mask.from_surface(base)
        outline = mask.to_surface(
            setcolor=(*color, 255),
            unsetcolor=(0, 0, 0, 0),
        ).convert_alpha()
        result = pygame.Surface(base.get_size(), pygame.SRCALPHA)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if dx * dx + dy * dy <= radius * radius + 1:
                    result.blit(outline, (dx, dy))
        result.blit(base, (0, 0))
        return result

    # -----------------------------------------------------------------
    # Menu screen
    # -----------------------------------------------------------------

    def _show_menu(self) -> None:
        buttons = [
            ("Player  vs  Player",  "pvp"),
            ("Player  vs  AI",      "pvai"),
            ("AI  vs  AI",          "aivai"),
            ("Replay",               "replay"),
        ]
        bw, bh = 340, 60
        spacing = 24
        total_h = len(buttons) * (bh + spacing) - spacing
        start_y = (self.H - total_h) // 2 + 60
        quit_rect = pygame.Rect((self.W - 220) // 2, self.H - 64, 220, 46)

        running = True
        while running:
            self.screen.fill(_C["bg"])
            # Title
            title = self.font_lg.render("Crazy House", True, _C["text"])
            self.screen.blit(title, title.get_rect(center=(self.W//2, self.H//4)))

            mx, my = self._to_logical(pygame.mouse.get_pos())
            btn_rects = []
            for i, (label, mode) in enumerate(buttons):
                bx = (self.W - bw) // 2
                by = start_y + i * (bh + spacing)
                rect = pygame.Rect(bx, by, bw, bh)
                btn_rects.append((rect, mode))
                hover = rect.collidepoint(mx, my)
                if mode == "replay":
                    color = (80, 170, 100) if hover else (55, 130, 75)
                else:
                    color = _C["btn_hover"] if hover else _C["btn"]
                pygame.draw.rect(self.screen, color, rect, border_radius=10)
                txt = self.font_md.render(label, True, _C["btn_txt"])
                self.screen.blit(txt, txt.get_rect(center=rect.center))

            self._draw_rect_button(quit_rect, "Quit", mx, my)

            self._flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if quit_rect.collidepoint(self._to_logical(ev.pos)):
                        pygame.quit(); sys.exit()
                    for rect, mode in btn_rects:
                        if rect.collidepoint(self._to_logical(ev.pos)):
                            if mode == "replay":
                                self._show_record_list()
                            else:
                                self._show_game_setup(mode)
                            return

    # -----------------------------------------------------------------
    # Game startup
    # -----------------------------------------------------------------

    def _show_game_setup(self, mode: str) -> None:
        bw, bh = 260, 46
        center_x = self.W // 2

        while True:
            self.screen.fill(_C["bg"])
            title = self.font_lg.render("Game Setup", True, _C["text"])
            self.screen.blit(title, title.get_rect(center=(center_x, 90)))

            mode_labels = {"pvp": "Player vs Player", "pvai": "Player vs AI",
                           "aivai": "AI vs AI"}
            mode_s = self.font_md.render(mode_labels.get(mode, ""), True, _C["text2"])
            self.screen.blit(mode_s, mode_s.get_rect(center=(center_x, 138)))

            mx, my = self._to_logical(pygame.mouse.get_pos())
            buttons: List[tuple[pygame.Rect, str]] = []

            y = 190
            self._draw_setup_label("Variant", y)
            buttons.extend(self._draw_choice_pair(
                y, "Crazy House", "drop_on", "Standard", "drop_off",
                self._setup_drop_mode,
            ))

            y += 74
            self._draw_setup_label("First move", y)
            buttons.extend(self._draw_choice_pair(
                y, "White", "first_white", "Black", "first_black",
                self._setup_first_color == Color.WHITE,
            ))

            y += 74
            if mode == "pvai":
                self._draw_setup_label("Human side", y)
                buttons.extend(self._draw_choice_pair(
                    y, "First", "human_first", "Second", "human_second",
                    self._setup_human_side == "first",
                ))
            elif mode == "pvp":
                self._draw_setup_label("Player 1", y)
                buttons.extend(self._draw_choice_pair(
                    y, "First", "human_first", "Second", "human_second",
                    self._setup_human_side == "first",
                ))

            y += 82
            self._draw_setup_stepper(
                "Main time (min)",
                y, "main_minus", "main_plus", buttons,
                input_action="main_input",
                value_text=str(self._setup_main_minutes),
                active=self._setup_active_field == "main",
                range_text="(0-60)",
            )

            y += 74
            self._draw_setup_stepper(
                "One-move time (sec)",
                y, "byo_minus", "byo_plus", buttons,
                input_action="byo_input",
                value_text=str(self._setup_byoyomi_seconds),
                active=self._setup_active_field == "byoyomi",
                range_text="(0-600)",
            )

            start_rect = pygame.Rect(center_x - bw - 12, y + 90, bw, bh + 6)
            back_rect = pygame.Rect(center_x + 12, y + 90, bw, bh + 6)
            buttons.extend([(start_rect, "start"), (back_rect, "back")])
            self._draw_rect_button(start_rect, "Start", mx, my)
            self._draw_rect_button(back_rect, "Back", mx, my)

            self._flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if self._handle_setup_key(ev):
                        continue
                    if ev.key == pygame.K_ESCAPE:
                        self._show_menu()
                        return
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    clicked_any = False
                    for rect, action in buttons:
                        if not rect.collidepoint(self._to_logical(ev.pos)):
                            continue
                        clicked_any = True
                        if action == "drop_on":
                            self._deactivate_setup_input()
                            self._setup_drop_mode = True
                        elif action == "drop_off":
                            self._deactivate_setup_input()
                            self._setup_drop_mode = False
                        elif action == "first_white":
                            self._deactivate_setup_input()
                            self._setup_first_color = Color.WHITE
                        elif action == "first_black":
                            self._deactivate_setup_input()
                            self._setup_first_color = Color.BLACK
                        elif action == "human_first":
                            self._deactivate_setup_input()
                            self._setup_human_side = "first"
                        elif action == "human_second":
                            self._deactivate_setup_input()
                            self._setup_human_side = "second"
                        elif action == "main_minus":
                            self._deactivate_setup_input()
                            self._setup_main_minutes = max(0, self._setup_main_minutes - 1)
                        elif action == "main_plus":
                            self._deactivate_setup_input()
                            self._setup_main_minutes = min(60, self._setup_main_minutes + 1)
                        elif action == "main_input":
                            self._activate_setup_input("main", self._setup_main_minutes)
                        elif action == "byo_minus":
                            self._deactivate_setup_input()
                            self._setup_byoyomi_seconds = max(0, self._setup_byoyomi_seconds - 1)
                        elif action == "byo_plus":
                            self._deactivate_setup_input()
                            self._setup_byoyomi_seconds = min(600, self._setup_byoyomi_seconds + 1)
                        elif action == "byo_input":
                            self._activate_setup_input("byoyomi", self._setup_byoyomi_seconds)
                        elif action == "start":
                            self._deactivate_setup_input()
                            self._start_game(mode)
                            return
                        elif action == "back":
                            self._deactivate_setup_input()
                            self._show_menu()
                            return
                        break
                    if not clicked_any:
                        self._deactivate_setup_input()

    def _draw_setup_label(self, label: str, y: int) -> None:
        s = self.font_md.render(label, True, _C["text"])
        self.screen.blit(s, (self.W // 2 - 270, y + 9))

    def _draw_choice_pair(
        self,
        y: int,
        left_label: str,
        left_action: str,
        right_label: str,
        right_action: str,
        left_selected: bool,
    ) -> List[tuple[pygame.Rect, str]]:
        mx, my = self._to_logical(pygame.mouse.get_pos())
        left = pygame.Rect(self.W // 2 - 50, y, 150, 46)
        right = pygame.Rect(self.W // 2 + 112, y, 150, 46)
        self._draw_rect_button(left, left_label, mx, my, selected=left_selected)
        self._draw_rect_button(right, right_label, mx, my, selected=not left_selected)
        return [(left, left_action), (right, right_action)]

    def _activate_setup_input(self, field: str, value: int) -> None:
        self._setup_active_field = field
        self._setup_input_text = str(value)

    def _deactivate_setup_input(self) -> None:
        if self._setup_active_field is not None:
            self._commit_setup_input()
        self._setup_active_field = None
        self._setup_input_text = ""

    def _commit_setup_input(self) -> None:
        if self._setup_active_field is None:
            return
        value = int(self._setup_input_text) if self._setup_input_text else 0
        if self._setup_active_field == "main":
            self._setup_main_minutes = max(0, min(60, value))
        elif self._setup_active_field == "byoyomi":
            self._setup_byoyomi_seconds = max(0, min(600, value))

    def _handle_setup_key(self, ev: pygame.event.Event) -> bool:
        if self._setup_active_field is None:
            return False

        if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
            self._deactivate_setup_input()
            return True
        if ev.key == pygame.K_ESCAPE:
            self._deactivate_setup_input()
            return True
        if ev.key == pygame.K_BACKSPACE:
            self._setup_input_text = self._setup_input_text[:-1]
            return True
        if ev.unicode and ev.unicode.isdigit():
            max_len = 2 if self._setup_active_field == "main" else 3
            if len(self._setup_input_text) < max_len:
                self._setup_input_text += ev.unicode
            return True
        return True

    def _draw_setup_stepper(
        self,
        label: str,
        y: int,
        minus_action: str,
        plus_action: str,
        buttons: List[tuple[pygame.Rect, str]],
        input_action: Optional[str] = None,
        value_text: Optional[str] = None,
        active: bool = False,
        range_text: Optional[str] = None,
    ) -> None:
        mx, my = self._to_logical(pygame.mouse.get_pos())
        label_s = self.font_md.render(label, True, _C["text"])
        label_x = self.W // 2 - 270
        self.screen.blit(label_s, (label_x, y + 2 if range_text else y + 8))
        if range_text:
            range_s = self.font_xs.render(range_text, True, _C["text2"])
            self.screen.blit(range_s, (label_x, y + 29))
        if input_action is not None:
            input_rect = pygame.Rect(self.W // 2 + 22, y, 86, 46)
            pygame.draw.rect(
                self.screen,
                (245, 245, 245) if active else (95, 95, 100),
                input_rect,
                border_radius=8,
            )
            pygame.draw.rect(
                self.screen,
                _C["hl_hand"] if active else _C["btn"],
                input_rect,
                width=2,
                border_radius=8,
            )
            display = self._setup_input_text if active else (value_text or "")
            value_s = self.font_md.render(display, True, (30, 30, 30) if active else _C["text"])
            self.screen.blit(value_s, value_s.get_rect(center=input_rect.center))
            buttons.append((input_rect, input_action))
            minus = pygame.Rect(self.W // 2 + 122, y, 48, 46)
            plus = pygame.Rect(self.W // 2 + 180, y, 48, 46)
        else:
            minus = pygame.Rect(self.W // 2 + 70, y, 54, 46)
            plus = pygame.Rect(self.W // 2 + 138, y, 54, 46)
        self._draw_rect_button(minus, "-", mx, my)
        self._draw_rect_button(plus, "+", mx, my)
        buttons.extend([(minus, minus_action), (plus, plus_action)])

    def _draw_rect_button(
        self,
        rect: pygame.Rect,
        label: str,
        mx: int,
        my: int,
        selected: bool = False,
        font: Optional[pygame.font.Font] = None,
        text_color: Optional[tuple[int, int, int]] = None,
    ) -> None:
        color = _C["hl_hand"] if selected else (_C["btn_hover"] if rect.collidepoint(mx, my) else _C["btn"])
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        text = (font or self.font_md).render(label, True, text_color or _C["btn_txt"])
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _start_game(self, mode: str) -> None:
        self.game_mode   = mode
        self.state       = GameState()
        self.state.drop_mode = self._setup_drop_mode
        self.state.current_player = self._setup_first_color
        self.state.position_history = [self.state._position_key()]
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
        self._front_winrate = None
        self._ai_candidates = None
        self._value_eval_dirty = True
        self._time_winner = None
        self._resign_winner = None
        self._is_replay = False
        self.first_color = self._setup_first_color
        self.main_time_sec = float(self._setup_main_minutes * 60)
        self.byoyomi_sec = float(self._setup_byoyomi_seconds)
        self._clock_remaining = {
            Color.WHITE: self.main_time_sec,
            Color.BLACK: self.main_time_sec,
        }
        self._turn_started_at = time.monotonic()

        first = self._setup_first_color
        second = first.opponent()
        self.human_color = first if self._setup_human_side == "first" else second
        self.player1_color = self.human_color

        if mode == "pvp":
            self.ai_white = None
            self.ai_black = None
        elif mode == "pvai":
            agent = self._get_shared_agent()
            self.ai_white = None if self.human_color == Color.WHITE else agent
            self.ai_black = None if self.human_color == Color.BLACK else agent
        elif mode == "aivai":
            agent = self._get_shared_agent()
            self.ai_white = agent
            self.ai_black = agent

        self._game_loop()

    def _get_shared_agent(self):
        if self._shared_agent is None:
            from ai.agent import AIAgent
            self._shared_agent = AIAgent(self.cfg, self.default_ckpt)
            self._value_eval_dirty = True
        return self._shared_agent

    # -----------------------------------------------------------------
    # Game loop
    # -----------------------------------------------------------------

    def _game_loop(self) -> None:
        while True:
            self.clock.tick(self.FPS)
            self._check_time_forfeit()
            self._handle_events()
            self._check_ai_result()
            self._maybe_start_ai()
            self._draw_frame()
            self._flip()

            if (self._time_winner is not None
                    or self._resign_winner is not None
                    or self.state.is_terminal()):
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
                    self._show_escape_dialog()
                    return

            if ev.type == pygame.VIDEORESIZE:
                pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)

            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if self._resign_btn_rect.collidepoint(self._to_logical(ev.pos)):
                        self._show_resign_dialog()
                        continue
                    if self._save_btn_rect.collidepoint(self._to_logical(ev.pos)):
                        self._show_save_record_dialog()
                        continue
                    self._on_click(self._to_logical(ev.pos))
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
            if color != self.state.current_player:
                self.status_msg = "Use current player's hand."
                self._deselect()
                return
            if self.state.hands[color].get(pt, 0) <= 0:
                self.status_msg = "No piece in hand."
                self._deselect()
                return
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
            self._flip()
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
        self._commit_elapsed_time(player)
        self.game_record.append(self._record_move(move, player, self.move_number + 1))
        self.state.apply_move(move)
        gives_check = self.state.is_in_check(self.state.current_player)
        self.game_record[-1]["check"] = gives_check
        self.last_move   = move
        self.move_number += 1
        self._value_eval_dirty = True
        self._ai_candidates = None
        self.status_msg = "CHECK!" if gives_check else ""
        self._turn_started_at = time.monotonic()
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
            "check": False,
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

    def _elapsed_current_turn(self) -> float:
        return max(0.0, time.monotonic() - self._turn_started_at)

    def _clock_snapshot(self, color: Color) -> tuple[float, float]:
        main = max(0.0, self._clock_remaining[color])
        byoyomi = float(self.byoyomi_sec)

        if (self.state is not None
                and not self._is_replay
                and self._time_winner is None
                and self._resign_winner is None
                and color == self.state.current_player):
            elapsed = self._elapsed_current_turn()
            if main > 0:
                if elapsed <= main:
                    return main - elapsed, byoyomi
                return 0.0, max(0.0, byoyomi - (elapsed - main))
            return 0.0, max(0.0, byoyomi - elapsed)

        return main, byoyomi

    def _commit_elapsed_time(self, color: Color) -> None:
        if (self._is_replay
                or self._time_winner is not None
                or self._resign_winner is not None):
            return

        elapsed = self._elapsed_current_turn()
        main = self._clock_remaining[color]
        if main > 0:
            spent_main = min(main, elapsed)
            self._clock_remaining[color] = max(0.0, main - spent_main)
            elapsed -= spent_main
        if self._clock_remaining[color] <= 0 and elapsed > self.byoyomi_sec:
            self._time_winner = color.opponent()

    def _check_time_forfeit(self) -> None:
        if (self._is_replay
                or self.state is None
                or self._time_winner is not None
                or self._resign_winner is not None):
            return
        main_left, move_left = self._clock_snapshot(self.state.current_player)
        if main_left <= 0 and move_left <= 0:
            self._time_winner = self.state.current_player.opponent()
            self.status_msg = "Time forfeiture"

    def _format_time(self, seconds: float) -> str:
        seconds = max(0.0, seconds)
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _maybe_start_ai(self) -> None:
        if (self._ai_busy
                or self.state.is_terminal()
                or self._time_winner is not None
                or self._resign_winner is not None):
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
        if self._time_winner is not None or self._resign_winner is not None:
            return
        if tag == "move":
            move: Move = payload
            # Small pause so AI moves are visible in AI vs AI
            if self.game_mode == "aivai":
                time.sleep(self.ANIM_MS / 1000.0)
            self._execute_move(move)
        else:
            self.status_msg = f"AI error: {payload}"

    def _get_front_winrate(self) -> Optional[float]:
        """Return the near-side (White) win probability in percent."""
        if self.state is None:
            return None

        if self._resign_winner is not None:
            self._front_winrate = 100.0 if self._resign_winner == Color.WHITE else 0.0
            self._ai_candidates = None
            self._value_eval_dirty = False
            return self._front_winrate

        if not self._value_eval_dirty:
            return self._front_winrate

        # Show 50% at move 0 (game start = equal position, no evaluation needed yet)
        if self.move_number == 0 and not self._is_replay:
            self._front_winrate = 50.0
            self._ai_candidates = None
            self._value_eval_dirty = False
            return 50.0

        if self.state.is_terminal():
            winner = self.state.get_winner()
            if winner == Color.WHITE:
                self._front_winrate = 100.0
            if winner == Color.BLACK:
                self._front_winrate = 0.0
            if winner is None:
                self._front_winrate = 50.0
            self._ai_candidates = None
            self._value_eval_dirty = False
            return self._front_winrate

        if self._shared_agent is None:
            self._front_winrate = None
            self._ai_candidates = None
            self._value_eval_dirty = False
            return None

        try:
            import torch
            from ai.network import state_to_tensor

            agent = self._shared_agent
            agent.network.eval()
            obs = state_to_tensor(self.state, agent.device)
            with torch.no_grad():
                policy_logits, value_t = agent.network(obs)
            current_value = float(value_t[0, 0].item())

            # The network value is from the side-to-move perspective.
            # The near side in this UI is White, so flip Black-to-move values.
            front_value = (current_value if self.state.current_player == Color.WHITE
                           else -current_value)
            self._front_winrate = max(0.0, min(100.0, (front_value + 1.0) * 50.0))

            # Compute top candidate moves from policy head (same forward pass)
            probs = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
            legal = self.state.get_legal_moves()
            if legal:
                legal_sorted = sorted(
                    legal,
                    key=lambda m: float(probs[move_to_action_idx(m)]),
                    reverse=True,
                )
                top = [str(m) for m in legal_sorted[:4]]
                self._ai_candidates = (top[0], top[1:]) if top else None
            else:
                self._ai_candidates = None
        except Exception as e:
            self._front_winrate = None
            self._ai_candidates = None
            self.status_msg = f"Value eval error: {e}"
        finally:
            self._value_eval_dirty = False

        return self._front_winrate

    def _get_ai_candidates(self) -> Optional[Tuple[str, List[str]]]:
        """Return cached (best_move_str, [next_strs]) from last policy evaluation."""
        return self._ai_candidates

    def _update_layout(self) -> None:
        pass  # layout is based on fixed logical dimensions; scaling done in _flip()

    def _flip(self) -> None:
        """Scale the logical surface to the actual window and present it."""
        window = pygame.display.get_surface()
        ww, wh = window.get_size()
        if (ww, wh) == (self.W, self.H):
            window.blit(self.screen, (0, 0))
        else:
            window.blit(pygame.transform.smoothscale(self.screen, (ww, wh)), (0, 0))
        pygame.display.flip()

    def _to_logical(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert a window-space coordinate to logical (drawing) space."""
        window = pygame.display.get_surface()
        ww, wh = window.get_size()
        if (ww, wh) == (self.W, self.H):
            return pos
        return (round(pos[0] * self.W / ww), round(pos[1] * self.H / wh))

    def _show_escape_dialog(self) -> None:
        """Popup 'Continue?' dialog; timers are paused while the popup is open."""
        popup_opened = time.monotonic()

        dialog   = pygame.Rect(0, 0, 380, 200)
        dialog.center = (self.W // 2, self.H // 2)
        quit_rect = pygame.Rect(dialog.centerx - 115, dialog.bottom - 64, 100, 40)
        yes_rect  = pygame.Rect(dialog.centerx + 15,  dialog.bottom - 64, 100, 40)

        while True:
            self._draw_frame()
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.screen.blit(overlay, (0, 0))

            pygame.draw.rect(self.screen, (245, 245, 245), dialog, border_radius=12)
            pygame.draw.rect(self.screen, (130, 130, 130), dialog, width=2, border_radius=12)
            title = self.font_lg.render("Continue?", True, (30, 30, 30))
            self.screen.blit(title, title.get_rect(center=(dialog.centerx, dialog.y + 68)))

            mx, my = self._to_logical(pygame.mouse.get_pos())
            self._draw_rect_button(quit_rect, "Quit", mx, my, font=self.font_sm,
                                   text_color=(220, 80, 80))
            self._draw_rect_button(yes_rect, "Yes", mx, my, font=self.font_sm)
            self._flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    self._turn_started_at += time.monotonic() - popup_opened
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if quit_rect.collidepoint(self._to_logical(ev.pos)):
                        self._turn_started_at += time.monotonic() - popup_opened
                        self._show_menu()
                        return
                    if yes_rect.collidepoint(self._to_logical(ev.pos)):
                        self._turn_started_at += time.monotonic() - popup_opened
                        return

            self.clock.tick(self.FPS)

    def _fit_text_tail(self, text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        fitted = text
        while fitted and font.size(fitted)[0] > max_width:
            fitted = fitted[1:]
        return fitted

    def _wrap_dialog_text(
        self,
        text: str,
        font: pygame.font.Font,
        max_width: int,
    ) -> list[str]:
        if not text:
            return [""]

        lines: list[str] = []
        current = ""
        for ch in text:
            if ch == "\n":
                lines.append(current)
                current = ""
                continue
            candidate = current + ch
            if current and font.size(candidate)[0] > max_width:
                lines.append(current)
                current = ch
            else:
                current = candidate
        lines.append(current)
        return lines

    def _draw_dialog_text_input(
        self,
        rect: pygame.Rect,
        label: str,
        text: str,
        active: bool,
        multiline: bool = False,
    ) -> None:
        label_s = self.font_sm.render(label, True, (70, 70, 70))
        self.screen.blit(label_s, (rect.x, rect.y - 26))

        pygame.draw.rect(self.screen, (250, 250, 250), rect, border_radius=6)
        border = (45, 105, 180) if active else (150, 150, 150)
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=6)

        display_text = text + ("|" if active else "")
        color = (25, 25, 25)
        if multiline:
            line_h = self.font_sm.get_linesize()
            max_lines = max(1, (rect.h - 14) // line_h)
            lines = self._wrap_dialog_text(display_text, self.font_sm, rect.w - 20)
            visible = lines[-max_lines:]
            y = rect.y + 8
            for line in visible:
                line_s = self.font_sm.render(line, True, color)
                self.screen.blit(line_s, (rect.x + 10, y))
                y += line_h
        else:
            fitted = self._fit_text_tail(display_text, self.font_md, rect.w - 20)
            text_s = self.font_md.render(fitted, True, color)
            self.screen.blit(text_s, text_s.get_rect(midleft=(rect.x + 10, rect.centery)))

    def _show_save_record_dialog(self) -> None:
        if self.state is None:
            return

        default_name = f"{self.game_mode or 'unknown'}_{self.move_number:03d}moves"
        save_name = default_name
        memo = ""
        active_field = "name"
        status_msg = ""
        input_limit = {"name": 80, "memo": 300}

        dialog = pygame.Rect(0, 0, 560, 380)
        dialog.center = (self.W // 2, self.H // 2)
        name_rect = pygame.Rect(dialog.x + 36, dialog.y + 96, dialog.w - 72, 42)
        memo_rect = pygame.Rect(dialog.x + 36, dialog.y + 178, dialog.w - 72, 84)
        save_rect = pygame.Rect(dialog.centerx - 170, dialog.bottom - 66, 150, 44)
        cancel_rect = pygame.Rect(dialog.centerx + 20, dialog.bottom - 66, 150, 44)
        complete_msg = "Save Completed!"

        def draw(status: str = "") -> None:
            self._draw_frame()
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 130))
            self.screen.blit(overlay, (0, 0))

            pygame.draw.rect(self.screen, (250, 250, 250), dialog, border_radius=10)
            pygame.draw.rect(self.screen, (120, 120, 120), dialog, width=2, border_radius=10)
            title_s = self.font_lg.render("Save Record", True, (25, 25, 25))
            self.screen.blit(title_s, title_s.get_rect(center=(dialog.centerx, dialog.y + 42)))

            self._draw_dialog_text_input(
                name_rect, "Save name", save_name, active_field == "name")
            self._draw_dialog_text_input(
                memo_rect, "Memo", memo, active_field == "memo", multiline=True)

            if status:
                status_color = (200, 40, 40) if status == complete_msg else (60, 60, 60)
                status_s = self.font_sm.render(status, True, status_color)
                self.screen.blit(status_s, status_s.get_rect(center=(dialog.centerx, dialog.bottom - 92)))

            mx, my = self._to_logical(pygame.mouse.get_pos())
            self._draw_rect_button(save_rect, "Save", mx, my, font=self.font_sm)
            self._draw_rect_button(cancel_rect, "Cancel", mx, my, font=self.font_sm)
            self._flip()

        def commit_save() -> bool:
            nonlocal status_msg
            draw("Saving...")
            try:
                self._save_game_record(save_name, memo)
            except Exception as e:
                status_msg = f"Save failed: {e}"
                return False

            draw(complete_msg)
            pygame.time.delay(800)
            return True

        try:
            while True:
                draw(status_msg)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit(); sys.exit()

                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            return
                        if ev.key == pygame.K_TAB:
                            active_field = "memo" if active_field == "name" else "name"
                            continue
                        if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            if active_field == "name":
                                active_field = "memo"
                            else:
                                if commit_save():
                                    return
                            continue
                        if ev.key == pygame.K_BACKSPACE:
                            if active_field == "name":
                                save_name = save_name[:-1]
                            else:
                                memo = memo[:-1]
                            continue

                        if ev.unicode and ev.unicode.isprintable():
                            if active_field == "name" and len(save_name) < input_limit["name"]:
                                save_name += ev.unicode
                            elif active_field == "memo" and len(memo) < input_limit["memo"]:
                                memo += ev.unicode

                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        if name_rect.collidepoint(self._to_logical(ev.pos)):
                            active_field = "name"
                        elif memo_rect.collidepoint(self._to_logical(ev.pos)):
                            active_field = "memo"
                        elif save_rect.collidepoint(self._to_logical(ev.pos)):
                            if commit_save():
                                return
                        elif cancel_rect.collidepoint(self._to_logical(ev.pos)):
                            return

                self.clock.tick(self.FPS)
        finally:
            self._turn_started_at = time.monotonic()

    def _show_resign_dialog(self) -> None:
        if self.state is None or self._is_replay or self._resign_winner is not None:
            return

        dialog = pygame.Rect(0, 0, 360, 180)
        dialog.center = (self.W // 2, self.H // 2)
        yes_rect = pygame.Rect(dialog.centerx - 115, dialog.bottom - 58, 90, 36)
        no_rect = pygame.Rect(dialog.centerx + 25, dialog.bottom - 58, 90, 36)

        try:
            while True:
                self._draw_frame()
                overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 135))
                self.screen.blit(overlay, (0, 0))

                pygame.draw.rect(self.screen, (250, 250, 250), dialog, border_radius=10)
                pygame.draw.rect(self.screen, (135, 135, 135), dialog, width=2, border_radius=10)
                title = self.font_lg.render("Really Quit?", True, (210, 30, 30))
                self.screen.blit(title, title.get_rect(center=(dialog.centerx, dialog.y + 62)))

                mx, my = self._to_logical(pygame.mouse.get_pos())
                self._draw_rect_button(yes_rect, "Yes", mx, my, font=self.font_sm)
                self._draw_rect_button(no_rect, "No", mx, my, font=self.font_sm)
                self._flip()

                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        return
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        if yes_rect.collidepoint(self._to_logical(ev.pos)):
                            resigned = self.state.current_player
                            self._commit_elapsed_time(resigned)
                            self._resign_winner = resigned.opponent()
                            self._value_eval_dirty = True
                            self.status_msg = (
                                "White resigned."
                                if resigned == Color.WHITE else "Black resigned."
                            )
                            self._deselect()
                            return
                        if no_rect.collidepoint(self._to_logical(ev.pos)):
                            return

                self.clock.tick(self.FPS)
        finally:
            self._turn_started_at = time.monotonic()

    def _safe_record_filename_part(self, text: str) -> str:
        illegal = set('<>:"/\\|?*')
        cleaned = []
        for ch in text.strip():
            if ch in illegal or ord(ch) < 32:
                cleaned.append("_")
            elif ch.isspace():
                cleaned.append("_")
            else:
                cleaned.append(ch)

        part = "".join(cleaned).strip(" ._")
        while "__" in part:
            part = part.replace("__", "_")
        return part[:64] or "record"

    def _unique_record_path(self, record_dir: str, stem: str) -> str:
        path = os.path.join(record_dir, f"{stem}.json")
        if not os.path.exists(path):
            return path

        idx = 2
        while True:
            candidate = os.path.join(record_dir, f"{stem}_{idx}.json")
            if not os.path.exists(candidate):
                return candidate
            idx += 1

    def _save_game_record(self, save_name: str = "", memo: str = "") -> Optional[str]:
        if self.state is None:
            return None

        record_dir = os.path.join(self.asset_dir, "gamerecord")
        os.makedirs(record_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = self.game_mode or "unknown"
        default_name = f"{mode}_{self.move_number:03d}moves"
        display_name = save_name.strip() or default_name
        filename_part = self._safe_record_filename_part(display_name)
        path = self._unique_record_path(record_dir, f"{timestamp}_{filename_part}")

        winner = self._resign_winner or self._time_winner or self.state.get_winner()
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
            "save_name": display_name,
            "memo": memo.strip(),
            "started_at": self._game_started_at,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "first_color": "white" if self.first_color == Color.WHITE else "black",
            "variant": "crazy_house" if self.state.drop_mode else "standard",
            "time_control": {
                "main_time_sec": self.main_time_sec,
                "byoyomi_sec": self.byoyomi_sec,
            },
            "move_count": self.move_number,
            "result": result,
            "ending": (
                "resign" if self._resign_winner is not None
                else "time" if self._time_winner is not None
                else "normal"
            ),
            "moves": self.game_record,
            "final_board": repr(self.state),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._last_saved_record = path
        self.status_msg = f"Saved: {os.path.basename(path)}"
        return path

    def _record_paths(self, record_dir: str) -> list[str]:
        return sorted(
            [
                p for p in (os.path.join(record_dir, name) for name in os.listdir(record_dir))
                if os.path.isfile(p) and p.lower().endswith(".json")
            ],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )

    def _delete_record_files(self, paths: list[str], record_dir: str) -> tuple[int, int]:
        root = os.path.normcase(os.path.abspath(record_dir))
        prefix = root + os.sep
        deleted = 0
        errors = 0

        for path in paths:
            target = os.path.normcase(os.path.abspath(path))
            if not target.startswith(prefix):
                errors += 1
                continue
            if not target.lower().endswith(".json"):
                errors += 1
                continue
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    deleted += 1
            except OSError:
                errors += 1

        return deleted, errors

    def _confirm_record_delete(self, count: int) -> bool:
        if count <= 0:
            return False

        base = self.screen.copy()
        dialog = pygame.Rect(0, 0, 470, 220)
        dialog.center = (self.W // 2, self.H // 2)
        delete_rect = pygame.Rect(dialog.centerx - 160, dialog.bottom - 66, 135, 42)
        cancel_rect = pygame.Rect(dialog.centerx + 25, dialog.bottom - 66, 135, 42)

        while True:
            self.screen.blit(base, (0, 0))
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 135))
            self.screen.blit(overlay, (0, 0))

            pygame.draw.rect(self.screen, (250, 250, 250), dialog, border_radius=10)
            pygame.draw.rect(self.screen, (135, 135, 135), dialog, width=2, border_radius=10)
            title = self.font_lg.render("Delete GameRecord?", True, (210, 30, 30))
            self.screen.blit(title, title.get_rect(center=(dialog.centerx, dialog.y + 52)))
            detail = self.font_sm.render(
                f"Delete {count} record(s). This cannot be undone.",
                True,
                (40, 40, 40),
            )
            self.screen.blit(detail, detail.get_rect(center=(dialog.centerx, dialog.y + 105)))

            mx, my = self._to_logical(pygame.mouse.get_pos())
            self._draw_rect_button(
                delete_rect, "Delete", mx, my, font=self.font_sm, text_color=(255, 220, 220))
            self._draw_rect_button(cancel_rect, "Cancel", mx, my, font=self.font_sm)
            self._flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    return False
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if delete_rect.collidepoint(self._to_logical(ev.pos)):
                        return True
                    if cancel_rect.collidepoint(self._to_logical(ev.pos)):
                        return False

            self.clock.tick(self.FPS)

    def _show_record_list(self) -> None:
        record_dir = os.path.join(self.asset_dir, "gamerecord")
        os.makedirs(record_dir, exist_ok=True)
        selected: set[str] = set()
        status_msg = ""

        while True:
            records = self._record_paths(record_dir)
            selected.intersection_update(records)

            self.screen.fill(_C["bg"])
            title = self.font_lg.render("Replay", True, _C["text"])
            self.screen.blit(title, title.get_rect(center=(self.W // 2, 70)))

            mx, my = self._to_logical(pygame.mouse.get_pos())
            buttons: List[tuple[pygame.Rect, str, Optional[str]]] = []
            y = 125
            row_h = 44
            if not records:
                msg = self.font_md.render("No saved records.", True, _C["text2"])
                self.screen.blit(msg, msg.get_rect(center=(self.W // 2, 210)))

            for path in records[:12]:
                check_rect = pygame.Rect(112, y + 5, 26, 26)
                row_rect = pygame.Rect(152, y, self.W - 500, row_h - 6)
                delete_rect = pygame.Rect(self.W - 320, y, 120, row_h - 6)
                mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                label = f"{mtime}  {os.path.basename(path)}"

                pygame.draw.rect(self.screen, (245, 245, 245), check_rect, border_radius=4)
                pygame.draw.rect(
                    self.screen,
                    _C["hl_hand"] if path in selected else (145, 145, 145),
                    check_rect,
                    width=2,
                    border_radius=4,
                )
                if path in selected:
                    mark = self.font_sm.render("X", True, (35, 35, 35))
                    self.screen.blit(mark, mark.get_rect(center=check_rect.center))

                display = self._fit_text_tail(label, self.font_sm, row_rect.w - 20)
                self._draw_rect_button(row_rect, display, mx, my, font=self.font_sm)
                self._draw_rect_button(
                    delete_rect, "Delete", mx, my, font=self.font_sm,
                    text_color=(255, 220, 220),
                )
                buttons.append((check_rect, "select", path))
                buttons.append((row_rect, "open", path))
                buttons.append((delete_rect, "delete_one", path))
                y += row_h

            if status_msg:
                status = self.font_sm.render(status_msg, True, (220, 150, 70))
                self.screen.blit(status, status.get_rect(center=(self.W // 2, self.H - 122)))

            delete_selected = pygame.Rect(150, self.H - 80, 260, 46)
            delete_all = pygame.Rect((self.W - 220) // 2, self.H - 80, 220, 46)
            back = pygame.Rect(self.W - 410, self.H - 80, 220, 46)
            self._draw_rect_button(
                delete_selected,
                f"Delete Selected ({len(selected)})",
                mx,
                my,
                font=self.font_sm,
                text_color=(255, 220, 220),
            )
            self._draw_rect_button(
                delete_all, "Delete All", mx, my, font=self.font_sm,
                text_color=(255, 220, 220))
            self._draw_rect_button(back, "Back", mx, my)
            buttons.append((delete_selected, "delete_selected", None))
            buttons.append((delete_all, "delete_all", None))
            buttons.append((back, "back", None))

            self._flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    self._show_menu()
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for rect, action, path in buttons:
                        if not rect.collidepoint(self._to_logical(ev.pos)):
                            continue
                        if action == "select" and path:
                            if path in selected:
                                selected.remove(path)
                            else:
                                selected.add(path)
                            break
                        if action == "open" and path:
                            self._load_replay_record(path)
                            self._replay_loop()
                            return
                        if action == "delete_one" and path:
                            if self._confirm_record_delete(1):
                                deleted, errors = self._delete_record_files([path], record_dir)
                                selected.discard(path)
                                status_msg = f"Deleted {deleted} record(s)."
                                if errors:
                                    status_msg += f" Failed: {errors}."
                            break
                        if action == "delete_selected":
                            if not selected:
                                status_msg = "No records selected."
                                break
                            targets = list(selected)
                            if self._confirm_record_delete(len(targets)):
                                deleted, errors = self._delete_record_files(targets, record_dir)
                                selected.clear()
                                status_msg = f"Deleted {deleted} record(s)."
                                if errors:
                                    status_msg += f" Failed: {errors}."
                            break
                        if action == "delete_all":
                            if not records:
                                status_msg = "No records to delete."
                                break
                            if self._confirm_record_delete(len(records)):
                                deleted, errors = self._delete_record_files(records, record_dir)
                                selected.clear()
                                status_msg = f"Deleted {deleted} record(s)."
                                if errors:
                                    status_msg += f" Failed: {errors}."
                            break
                        if action == "back":
                            self._show_menu()
                            return

    def _load_replay_record(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._is_replay = True
        self.game_mode = "replay"
        self.ai_white = None
        self.ai_black = None
        self._ai_busy = False
        self._time_winner = None
        self._resign_winner = None
        self._replay_auto = False
        self._replay_last_step = time.monotonic()
        self._replay_moves = list(data.get("moves", []))
        self._replay_source_name = os.path.basename(path)
        self.first_color = Color.BLACK if data.get("first_color") == "black" else Color.WHITE
        self._setup_drop_mode = (data.get("variant", "crazy_house") != "standard")
        self.status_msg = ""
        self._set_replay_index(0)

    def _move_from_record(self, rec: dict) -> Move:
        promotion = PieceType[rec["promotion"]] if rec.get("promotion") else None
        drop_piece = PieceType[rec["drop_piece"]] if rec.get("drop_piece") else None
        from_pos = tuple(rec["from"]) if rec.get("from") is not None else None
        to_pos = tuple(rec["to"])
        return Move(
            from_pos=from_pos,
            to_pos=to_pos,
            promotion=promotion,
            is_drop=bool(rec.get("is_drop", False)),
            drop_piece=drop_piece,
        )

    def _set_replay_index(self, index: int) -> None:
        self._replay_index = max(0, min(index, len(self._replay_moves)))
        state = GameState()
        state.drop_mode = self._setup_drop_mode
        state.current_player = self.first_color
        state.position_history = [state._position_key()]
        last_move = None
        for rec in self._replay_moves[:self._replay_index]:
            move = self._move_from_record(rec)
            state.apply_move(move)
            last_move = move
        self.state = state
        self.last_move = last_move
        self.move_number = self._replay_index
        self.selected_square = None
        self.selected_hand = None
        self.highlight_squares = []
        self._front_winrate = None
        self._ai_candidates = None
        self._value_eval_dirty = True

    def _replay_buttons(self) -> List[tuple[pygame.Rect, str, str]]:
        x = 8
        y = 210
        w = 25
        h = 34
        gap = 3
        play_label = "||" if self._replay_auto else "Go"
        items = [
            ("<<", "start"),
            ("<", "prev"),
            (play_label, "play"),
            (">", "next"),
            (">>", "end"),
        ]
        return [
            (pygame.Rect(x + i * (w + gap), y, w, h), label, action)
            for i, (label, action) in enumerate(items)
        ]

    def _replay_loop(self) -> None:
        while self._is_replay:
            self.clock.tick(self.FPS)
            now = time.monotonic()
            if self._replay_auto and now - self._replay_last_step >= 1.0:
                if self._replay_index < len(self._replay_moves):
                    self._set_replay_index(self._replay_index + 1)
                    self._replay_last_step = now
                else:
                    self._replay_auto = False

            self._handle_replay_events()
            self._draw_frame()
            self._flip()

    def _handle_replay_events(self) -> None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self._replay_auto = False
                self._is_replay = False
                self._show_menu()
                return
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for rect, _label, action in self._replay_buttons():
                    if not rect.collidepoint(self._to_logical(ev.pos)):
                        continue
                    if action == "start":
                        self._replay_auto = False
                        self._set_replay_index(0)
                    elif action == "prev":
                        self._replay_auto = False
                        self._set_replay_index(self._replay_index - 1)
                    elif action == "play":
                        self._replay_auto = not self._replay_auto
                        self._replay_last_step = time.monotonic()
                    elif action == "next":
                        self._replay_auto = False
                        self._set_replay_index(self._replay_index + 1)
                    elif action == "end":
                        self._replay_auto = False
                        self._set_replay_index(len(self._replay_moves))
                    return

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

            mx, my  = self._to_logical(pygame.mouse.get_pos())
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

            self._flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for rect, pt in btn_rects:
                        if rect.collidepoint(self._to_logical(ev.pos)):
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
        if self._is_replay:
            self._draw_replay_info_panel()
        else:
            self._draw_info_panel()
        self._draw_board()
        self._draw_pieces()
        self._draw_highlights()
        if self.state.drop_mode:
            self._draw_hand_panel(Color.BLACK)
            self._draw_hand_panel(Color.WHITE)

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

        section_y = self.BY if color == Color.BLACK else self.BY + self.BH // 2
        section_h = self.BH // 2
        is_current = (
            not self._is_replay
            and self.state is not None
            and self.state.current_player == color
            and self._time_winner is None
            and self._resign_winner is None
        )
        if is_current:
            section_rect = pygame.Rect(
                self.HAND_X + 4,
                section_y + 4,
                self.W - self.HAND_X - 12,
                section_h - 8,
            )
            pygame.draw.rect(self.screen, _C["hl_hand"], section_rect, width=2, border_radius=8)

        # Section title
        label = "White's hand" if color == Color.WHITE else "Black's hand"
        lbl_s = self.font_sm.render(label, True, _C["text"] if is_current else _C["text2"])
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
        # Turn indicator: fixed "Turn:" label + colored player name side-by-side
        turn_label_s = self.font_md.render("Turn:", True, _C["text"])
        self.screen.blit(turn_label_s, (10, y))
        turn_name = "White" if self.state.current_player == Color.WHITE else "Black"
        turn_col = (230, 230, 230) if self.state.current_player == Color.WHITE else (100, 140, 220)
        turn_name_s = self.font_md.render(turn_name, True, turn_col)
        self.screen.blit(turn_name_s, (10 + turn_label_s.get_width(), y))
        y += 36

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

        winrate = self._get_front_winrate()
        bar_x = 10
        bar_w = max(60, self.BX - 30)

        # Win% labels: upper-left = Black's win%, upper-right = White's win%
        if winrate is not None:
            b_lbl = self.font_xs.render(f"{100 - winrate:.0f}%", True, _C["text2"])
            w_lbl = self.font_xs.render(f"{winrate:.0f}%",       True, _C["text2"])
        else:
            b_lbl = self.font_xs.render("--", True, _C["text2"])
            w_lbl = self.font_xs.render("--", True, _C["text2"])
        self.screen.blit(b_lbl, (bar_x, y))
        self.screen.blit(w_lbl, (bar_x + bar_w - w_lbl.get_width(), y))
        y += b_lbl.get_height() + 2

        # Win gauge bar: left = Black (dark), right = White (light)
        bar_h = 12
        pygame.draw.rect(self.screen, (20, 20, 20), (bar_x, y, bar_w, bar_h), border_radius=3)
        if winrate is not None:
            white_w = max(0, int(bar_w * winrate / 100.0))
            black_w = bar_w - white_w
            if black_w > 0:
                pygame.draw.rect(self.screen, (40, 40, 40),
                                 (bar_x, y, black_w, bar_h), border_radius=3)
            if white_w > 0:
                pygame.draw.rect(self.screen, (215, 215, 215),
                                 (bar_x + black_w, y, white_w, bar_h), border_radius=3)
        pygame.draw.rect(self.screen, _C["text2"], (bar_x, y, bar_w, bar_h),
                         width=1, border_radius=3)
        y += bar_h + 6

        # AI candidate moves from policy head
        candidates = self._get_ai_candidates()
        if candidates is not None:
            best, others = candidates
            best_s = self.font_xs.render(f"Best: {best}", True, _C["text"])
            self.screen.blit(best_s, (10, y)); y += 18
            if others:
                alts = ", ".join(str(o) for o in others[:3])
                max_w = self.BX - 22
                while alts and self.font_xs.size(alts)[0] > max_w:
                    alts = alts.rsplit(",", 1)[0]
                alts_s = self.font_xs.render(alts, True, _C["text2"])
                self.screen.blit(alts_s, (10, y)); y += 18
        y += 4

        for color, name in ((Color.WHITE, "White"), (Color.BLACK, "Black")):
            main_left, move_left = self._clock_snapshot(color)
            active = (
                self.state.current_player == color
                and self._time_winner is None
                and self._resign_winner is None
            )
            side = "First" if color == self.first_color else "Second"
            label_col = _C["text"] if active else _C["text2"]
            line1 = self.font_xs.render(f"{name} ({side})", True, label_col)
            line2 = self.font_xs.render(
                f"Main {self._format_time(main_left)}  Move {move_left:04.1f}s",
                True,
                label_col,
            )
            self.screen.blit(line1, (10, y)); y += 18
            self.screen.blit(line2, (10, y)); y += 20
        y += 6

        # Check indicator
        if self.state.is_in_check(self.state.current_player):
            chk = self.font_md.render("CHECK!", True, _C["hl_check"])
            self.screen.blit(chk, (10, y)); y += 32

        # Resign / save buttons
        mx, my = self._to_logical(pygame.mouse.get_pos())
        resign_hover = self._resign_btn_rect.collidepoint(mx, my)
        pygame.draw.rect(self.screen, _C["btn_hover"] if resign_hover else _C["btn"],
                         self._resign_btn_rect, border_radius=6)
        resign_s = self.font_sm.render("Resign", True, _C["btn_txt"])
        self.screen.blit(resign_s, resign_s.get_rect(center=self._resign_btn_rect.center))

        save_hover = self._save_btn_rect.collidepoint(mx, my)
        pygame.draw.rect(self.screen, _C["btn_hover"] if save_hover else _C["btn"],
                         self._save_btn_rect, border_radius=6)
        save_s = self.font_sm.render("Save Record", True, _C["btn_txt"])
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

    def _draw_replay_info_panel(self) -> None:
        panel = pygame.Rect(0, 0, self.BX - 4, self.H)
        pygame.draw.rect(self.screen, _C["panel"], panel)

        y = 26
        title = self.font_md.render("Replay", True, _C["text"])
        self.screen.blit(title, (10, y)); y += 34

        source = self._replay_source_name
        if len(source) > 18:
            source = source[:15] + "..."
        src = self.font_xs.render(source, True, _C["text2"])
        self.screen.blit(src, (10, y)); y += 32

        progress = self.font_sm.render(
            f"Move {self._replay_index}/{len(self._replay_moves)}",
            True,
            _C["text"],
        )
        self.screen.blit(progress, (10, y)); y += 34

        mx, my = self._to_logical(pygame.mouse.get_pos())
        for rect, label, action in self._replay_buttons():
            text_color = (220, 30, 30) if action == "play" and label == "Go" else None
            self._draw_rect_button(rect, label, mx, my, font=self.font_sm, text_color=text_color)

        esc_s = self.font_sm.render("[ESC] Top", True, _C["text2"])
        self.screen.blit(esc_s, (10, self.H - 56))

    # -----------------------------------------------------------------
    # Game-over screen
    # -----------------------------------------------------------------

    def _show_game_over(self) -> None:
        if self._resign_winner is not None:
            headline = "Resign"
            detail = "White wins" if self._resign_winner == Color.WHITE else "Black wins"
        elif self._time_winner is not None:
            headline = "Time Up"
            detail = "White wins" if self._time_winner == Color.WHITE else "Black wins"
        elif self.state.is_checkmate():
            headline = "Check Mate"
            winner = self.state.get_winner()
            detail = "White wins" if winner == Color.WHITE else "Black wins"
        elif self.state.is_stalemate():
            headline = "Stale Mate"
            detail = "Draw"
        else:
            headline = "Draw"
            detail = "Draw"

        box = pygame.Rect(self.BX + 135, self.BY + 140, self.BW - 270, 310)
        bw, bh = 145, 38

        while True:
            self._draw_frame()

            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 95))
            self.screen.blit(overlay, (0, 0))

            pygame.draw.rect(self.screen, (250, 250, 245), box, border_radius=8)
            pygame.draw.rect(self.screen, (220, 220, 210), box, width=2, border_radius=8)

            ht = self.font_lg.render(headline, True, (210, 30, 30))
            self.screen.blit(ht, ht.get_rect(center=(box.centerx, box.y + 76)))
            dt = self.font_md.render(detail, True, (35, 35, 35))
            self.screen.blit(dt, dt.get_rect(center=(box.centerx, box.y + 122)))

            mx, my  = self._to_logical(pygame.mouse.get_pos())
            buttons = [
                (pygame.Rect(box.centerx - bw - 12, box.bottom - 118, bw, bh), "Replay"),
                (pygame.Rect(box.centerx + 12,      box.bottom - 118, bw, bh), "Top"),
                (pygame.Rect(box.centerx - 190,     box.bottom - 68, 380, bh), "Save Record"),
            ]
            for rect, label in buttons:
                hover = rect.collidepoint(mx, my)
                pygame.draw.rect(self.screen, _C["btn_hover"] if hover else _C["btn"],
                                 rect, border_radius=7)
                lbl = self.font_sm.render(label, True, _C["btn_txt"])
                self.screen.blit(lbl, lbl.get_rect(center=rect.center))

            if self.status_msg:
                sm = self.font_sm.render(self.status_msg, True, (80, 80, 80))
                self.screen.blit(sm, sm.get_rect(center=(box.centerx, box.bottom - 18)))

            self._flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for rect, label in buttons:
                        if rect.collidepoint(self._to_logical(ev.pos)):
                            if label == "Replay":
                                self._start_game(self.game_mode)
                                return
                            if label == "Top":
                                self._show_menu()
                                return
                            if label == "Save Record":
                                self._show_save_record_dialog()
