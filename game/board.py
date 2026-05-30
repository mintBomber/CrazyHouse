"""
GameState: full chess-shogi hybrid board state and game logic.

Variant rules vs standard chess:
  1. Captured pieces go to the capturer's hand and can be dropped on any empty
     square as the player's own piece (shogi-style).
  2. A pawn may NOT be dropped on the opponent's back rank (rank 8 for White,
     rank 1 for Black).
  3. A pawn drop that immediately checkmates the opponent is illegal.
  4. Kings are never captured; game ends by checkmate.

Board coordinate convention:
  row 0 = rank 8 (Black's back rank), row 7 = rank 1 (White's back rank)
  col 0 = file a, col 7 = file h
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict

from .pieces import PieceType, Color, Move, DROPPABLE_PIECES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INITIAL_BOARD = np.array([
    [-2, -3, -4, -5, -6, -4, -3, -2],   # row 0: Black back rank
    [-1, -1, -1, -1, -1, -1, -1, -1],   # row 1: Black pawns
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 1,  1,  1,  1,  1,  1,  1,  1],   # row 6: White pawns
    [ 2,  3,  4,  5,  6,  4,  3,  2],   # row 7: White back rank
], dtype=np.int8)

_KNIGHT_DELTAS = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
_ROOK_DIRS    = [(0,1),(0,-1),(1,0),(-1,0)]
_BISHOP_DIRS  = [(1,1),(1,-1),(-1,1),(-1,-1)]
_QUEEN_DIRS   = _ROOK_DIRS + _BISHOP_DIRS


class GameState:
    """
    Complete mutable state of one game.
    Call copy() before applying moves if you need the original state preserved.
    """

    __slots__ = (
        "board", "hands", "castling_rights", "en_passant",
        "current_player", "halfmove_clock", "fullmove_number",
        "position_history", "drop_mode",
    )

    def __init__(self) -> None:
        self.board: np.ndarray = _INITIAL_BOARD.copy()

        # hand[color][piece_type] = count
        self.hands: Dict[Color, Dict[PieceType, int]] = {
            Color.WHITE: {pt: 0 for pt in DROPPABLE_PIECES},
            Color.BLACK: {pt: 0 for pt in DROPPABLE_PIECES},
        }

        # Castling rights: (white_K, white_Q, black_K, black_Q)
        self.castling_rights: List[bool] = [True, True, True, True]
        # Indices: 0=white_K, 1=white_Q, 2=black_K, 3=black_Q

        self.en_passant: Optional[Tuple[int,int]] = None
        self.current_player: Color = Color.WHITE
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1

        # True = Crazy House (captures go to hand, drops allowed)
        # False = Standard Chess (no drops)
        self.drop_mode: bool = True

        # List of position keys for repetition detection
        self.position_history: List[str] = [self._position_key()]

    # ------------------------------------------------------------------
    # Copy / state key
    # ------------------------------------------------------------------

    def copy(self) -> "GameState":
        s = object.__new__(GameState)
        s.board = self.board.copy()
        s.hands = {
            Color.WHITE: dict(self.hands[Color.WHITE]),
            Color.BLACK: dict(self.hands[Color.BLACK]),
        }
        s.castling_rights = list(self.castling_rights)
        s.en_passant = self.en_passant
        s.current_player = self.current_player
        s.halfmove_clock = self.halfmove_clock
        s.fullmove_number = self.fullmove_number
        s.position_history = list(self.position_history)
        s.drop_mode = self.drop_mode
        return s

    def _position_key(self) -> str:
        """A hashable string that uniquely identifies the position (for repetition)."""
        bkey = self.board.tobytes()
        wh   = tuple(sorted(self.hands[Color.WHITE].items()))
        bh   = tuple(sorted(self.hands[Color.BLACK].items()))
        return (
            f"{bkey}|{wh}|{bh}|{self.castling_rights}|"
            f"{self.en_passant}|{int(self.current_player)}"
        )

    # ------------------------------------------------------------------
    # Check detection  (does not depend on get_legal_moves)
    # ------------------------------------------------------------------

    def is_in_check(self, color: Color) -> bool:
        """True if the king of *color* is attacked by any opponent piece."""
        king_val = int(PieceType.KING) * int(color)
        opp      = color.opponent()
        opp_sign = int(opp)

        # Locate king
        pos = np.argwhere(self.board == king_val)
        if len(pos) == 0:
            return True     # should never happen in a legal game
        kr, kc = int(pos[0,0]), int(pos[0,1])

        # Knight attacks
        nv = int(PieceType.KNIGHT) * opp_sign
        for dr, dc in _KNIGHT_DELTAS:
            r, c = kr+dr, kc+dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r,c] == nv:
                return True

        # Pawn attacks
        # Black pawn at (r,c) attacks (r+1, c±1); white pawn attacks (r-1, c±1).
        # To find the attacker: look at row kr-1 for the white king (black pawns above it),
        # and row kr+1 for the black king (white pawns below it).
        pawn_dir = -1 if color == Color.WHITE else 1
        pv = int(PieceType.PAWN) * opp_sign
        for dc in (-1, 1):
            r, c = kr+pawn_dir, kc+dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r,c] == pv:
                return True

        # Rook / Queen (straight lines)
        for dr, dc in _ROOK_DIRS:
            r, c = kr+dr, kc+dc
            while 0 <= r < 8 and 0 <= c < 8:
                t = self.board[r,c]
                if t:
                    if t == int(PieceType.ROOK)*opp_sign or t == int(PieceType.QUEEN)*opp_sign:
                        return True
                    break
                r += dr; c += dc

        # Bishop / Queen (diagonals)
        for dr, dc in _BISHOP_DIRS:
            r, c = kr+dr, kc+dc
            while 0 <= r < 8 and 0 <= c < 8:
                t = self.board[r,c]
                if t:
                    if t == int(PieceType.BISHOP)*opp_sign or t == int(PieceType.QUEEN)*opp_sign:
                        return True
                    break
                r += dr; c += dc

        # King (adjacent)
        kv_opp = int(PieceType.KING) * opp_sign
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0:
                    continue
                r, c = kr+dr, kc+dc
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r,c] == kv_opp:
                    return True

        return False

    # ------------------------------------------------------------------
    # Pseudo-legal move generation
    # ------------------------------------------------------------------

    def _can_move_to_target(self, target: int, player: Color) -> bool:
        if target == 0:
            return True
        if (target > 0) == (int(player) > 0):
            return False
        return abs(target) != int(PieceType.KING)

    def _pawn_moves(self, row: int, col: int, player: Color) -> List[Move]:
        moves: List[Move] = []
        fwd  = -1 if player == Color.WHITE else 1
        promo_row  = 0  if player == Color.WHITE else 7
        start_row  = 6  if player == Color.WHITE else 1

        nr = row + fwd
        if 0 <= nr < 8:
            # Forward single
            if self.board[nr, col] == 0:
                if nr == promo_row:
                    for pt in (PieceType.QUEEN, PieceType.ROOK,
                               PieceType.BISHOP, PieceType.KNIGHT):
                        moves.append(Move((row,col),(nr,col), promotion=pt))
                else:
                    moves.append(Move((row,col),(nr,col)))
                # Double push from starting row
                if row == start_row:
                    nr2 = row + 2*fwd
                    if 0 <= nr2 < 8 and self.board[nr2, col] == 0:
                        moves.append(Move((row,col),(nr2,col)))
            # Captures
            for dc in (-1, 1):
                nc = col + dc
                if 0 <= nc < 8:
                    t = self.board[nr, nc]
                    is_ep = (nr,nc) == self.en_passant
                    if ((t != 0 and self._can_move_to_target(int(t), player))
                            or is_ep):
                        if nr == promo_row:
                            for pt in (PieceType.QUEEN, PieceType.ROOK,
                                       PieceType.BISHOP, PieceType.KNIGHT):
                                moves.append(Move((row,col),(nr,nc), promotion=pt))
                        else:
                            moves.append(Move((row,col),(nr,nc)))
        return moves

    def _sliding_moves(self, row: int, col: int, player: Color,
                       dirs: List[Tuple[int,int]]) -> List[Move]:
        moves: List[Move] = []
        for dr, dc in dirs:
            r, c = row+dr, col+dc
            while 0 <= r < 8 and 0 <= c < 8:
                t = self.board[r, c]
                if t == 0:
                    moves.append(Move((row,col),(r,c)))
                else:
                    if self._can_move_to_target(int(t), player):
                        moves.append(Move((row,col),(r,c)))
                    break
                r += dr; c += dc
        return moves

    def _knight_moves(self, row: int, col: int, player: Color) -> List[Move]:
        moves: List[Move] = []
        for dr, dc in _KNIGHT_DELTAS:
            r, c = row+dr, col+dc
            if 0 <= r < 8 and 0 <= c < 8:
                t = self.board[r, c]
                if self._can_move_to_target(int(t), player):
                    moves.append(Move((row,col),(r,c)))
        return moves

    def _king_moves(self, row: int, col: int, player: Color) -> List[Move]:
        moves: List[Move] = []
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row+dr, col+dc
                if 0 <= r < 8 and 0 <= c < 8:
                    t = self.board[r, c]
                    if self._can_move_to_target(int(t), player):
                        moves.append(Move((row,col),(r,c)))

        # Castling (pseudo-legal; legality checked by is_in_check filtering)
        if player == Color.WHITE and row == 7 and col == 4:
            if (self.castling_rights[0]
                    and self.board[7,5] == 0 and self.board[7,6] == 0
                    and self.board[7,7] == int(PieceType.ROOK)):
                moves.append(Move((7,4),(7,6)))
            if (self.castling_rights[1]
                    and self.board[7,3] == 0 and self.board[7,2] == 0
                    and self.board[7,1] == 0
                    and self.board[7,0] == int(PieceType.ROOK)):
                moves.append(Move((7,4),(7,2)))
        elif player == Color.BLACK and row == 0 and col == 4:
            if (self.castling_rights[2]
                    and self.board[0,5] == 0 and self.board[0,6] == 0
                    and self.board[0,7] == -int(PieceType.ROOK)):
                moves.append(Move((0,4),(0,6)))
            if (self.castling_rights[3]
                    and self.board[0,3] == 0 and self.board[0,2] == 0
                    and self.board[0,1] == 0
                    and self.board[0,0] == -int(PieceType.ROOK)):
                moves.append(Move((0,4),(0,2)))
        return moves

    def _pseudo_board_moves(self) -> List[Move]:
        moves: List[Move] = []
        player      = self.current_player
        player_sign = int(player)
        for r in range(8):
            for c in range(8):
                p = int(self.board[r,c])
                if p == 0 or (p > 0) != (player_sign > 0):
                    continue
                pt = PieceType(abs(p))
                if pt == PieceType.PAWN:
                    moves.extend(self._pawn_moves(r, c, player))
                elif pt == PieceType.ROOK:
                    moves.extend(self._sliding_moves(r, c, player, _ROOK_DIRS))
                elif pt == PieceType.KNIGHT:
                    moves.extend(self._knight_moves(r, c, player))
                elif pt == PieceType.BISHOP:
                    moves.extend(self._sliding_moves(r, c, player, _BISHOP_DIRS))
                elif pt == PieceType.QUEEN:
                    moves.extend(self._sliding_moves(r, c, player, _QUEEN_DIRS))
                elif pt == PieceType.KING:
                    moves.extend(self._king_moves(r, c, player))
        return moves

    def _pseudo_drop_moves(self, apply_drop_mate_filter: bool = True) -> List[Move]:
        """Generate all pseudo-legal drop moves for the current player."""
        if not self.drop_mode:
            return []
        player     = self.current_player
        hand       = self.hands[player]
        back_rank  = 0 if player == Color.WHITE else 7   # opponent's back rank
        drop_moves: List[Move] = []

        for pt in DROPPABLE_PIECES:
            if hand[pt] <= 0:
                continue
            for r in range(8):
                for c in range(8):
                    if self.board[r,c] != 0:
                        continue
                    # Pawn cannot be dropped on opponent's back rank
                    if pt == PieceType.PAWN and r == back_rank:
                        continue
                    drop_moves.append(Move(None,(r,c), is_drop=True, drop_piece=pt))

        if not apply_drop_mate_filter:
            return drop_moves

        # Variant rule: only pawn drops that immediately checkmate are illegal.
        # _get_legal_moves_base disables this filter to avoid recursive mate checks.
        opp   = player.opponent()
        legal: List[Move] = []
        for mv in drop_moves:
            if mv.drop_piece != PieceType.PAWN:
                legal.append(mv)
                continue
            test = self.copy()
            test._apply_unchecked(mv)
            if test.is_in_check(opp):
                test.current_player = opp
                replies = test._get_legal_moves_base()
                if not replies:
                    continue   # Illegal: pawn drop causes immediate checkmate
            legal.append(mv)
        return legal

    # ------------------------------------------------------------------
    # Legal move generation
    # ------------------------------------------------------------------

    def _get_legal_moves_base(self) -> List[Move]:
        """
        Legal moves WITHOUT the drop-checkmate filter.
        Used internally to avoid infinite recursion in drop-mate checks.
        """
        player    = self.current_player
        pseudo    = self._pseudo_board_moves() + self._pseudo_drop_moves(
            apply_drop_mate_filter=False)
        legal: List[Move] = []

        for mv in pseudo:
            test = self.copy()
            test._apply_unchecked(mv)
            if not test.is_in_check(player):
                # Castling path check: king must not pass through an attacked square.
                # Only applies when the KING moves exactly 2 squares horizontally.
                if (not mv.is_drop
                        and mv.from_pos is not None
                        and mv.from_pos[0] == mv.to_pos[0]
                        and abs(mv.from_pos[1] - mv.to_pos[1]) == 2
                        and abs(int(self.board[mv.from_pos])) == int(PieceType.KING)):
                    if not self._castling_path_safe(mv, player):
                        continue
                legal.append(mv)
        return legal

    def get_legal_moves(self) -> List[Move]:
        """All legal moves for the current player including variant drop rules."""
        player    = self.current_player
        pseudo    = self._pseudo_board_moves() + self._pseudo_drop_moves(
            apply_drop_mate_filter=True)
        legal: List[Move] = []

        for mv in pseudo:
            test = self.copy()
            test._apply_unchecked(mv)
            if not test.is_in_check(player):
                if (not mv.is_drop
                        and mv.from_pos is not None
                        and mv.from_pos[0] == mv.to_pos[0]
                        and abs(mv.from_pos[1] - mv.to_pos[1]) == 2
                        and abs(int(self.board[mv.from_pos])) == int(PieceType.KING)):
                    if not self._castling_path_safe(mv, player):
                        continue
                legal.append(mv)
        return legal

    def _castling_path_safe(self, castle_move: Move, player: Color) -> bool:
        """
        Returns True if the castling path is safe (king not in check at start,
        not passing through an attacked square, and not landing in check).
        The destination check is handled by the caller's is_in_check test.
        """
        if self.is_in_check(player):
            return False
        fr, fc = castle_move.from_pos
        tc     = castle_move.to_pos[1]
        mid_c  = (fc + tc) // 2   # square the king passes through

        test = self.copy()
        piece = test.board[fr, fc]
        test.board[fr, mid_c] = piece
        test.board[fr, fc]    = 0
        return not test.is_in_check(player)

    # ------------------------------------------------------------------
    # Move application
    # ------------------------------------------------------------------

    def _apply_unchecked(self, move: Move) -> None:
        """Apply move without legality validation (modifies self in-place)."""
        player  = self.current_player
        opp     = player.opponent()
        sign    = int(player)
        is_pawn_or_capture = False

        if move.is_drop:
            val = int(move.drop_piece) * sign
            self.board[move.to_pos] = val
            self.hands[player][move.drop_piece] -= 1
            is_pawn_or_capture = (move.drop_piece == PieceType.PAWN)
            self.en_passant = None

        else:
            fr, fc = move.from_pos
            tr, tc = move.to_pos
            piece  = int(self.board[fr, fc])
            pt     = PieceType(abs(piece))

            prev_ep = self.en_passant
            self.en_passant = None   # reset; new EP target set below if applicable

            # En-passant capture: remove the captured pawn
            if pt == PieceType.PAWN and (tr, tc) == prev_ep:
                cap_r = tr + (1 if player == Color.WHITE else -1)
                captured_val = int(self.board[cap_r, tc])
                self.board[cap_r, tc] = 0
                if self.drop_mode:
                    self.hands[player][PieceType(abs(captured_val))] += 1
                is_pawn_or_capture = True

            # Regular capture
            target = int(self.board[tr, tc])
            if target != 0:
                cap_pt = PieceType(abs(target))
                if self.drop_mode:
                    self.hands[player][cap_pt] += 1
                is_pawn_or_capture = True
                # Losing castling rights when the rook square is captured
                if (tr, tc) == (7, 0): self.castling_rights[1] = False
                elif (tr, tc) == (7, 7): self.castling_rights[0] = False
                elif (tr, tc) == (0, 0): self.castling_rights[3] = False
                elif (tr, tc) == (0, 7): self.castling_rights[2] = False

            # Place piece
            self.board[tr, tc] = piece
            self.board[fr, fc] = 0

            # Set new en-passant target
            if pt == PieceType.PAWN:
                is_pawn_or_capture = True
                if abs(tr - fr) == 2:
                    self.en_passant = ((fr + tr) // 2, tc)

            # Promotion
            if pt == PieceType.PAWN and move.promotion:
                self.board[tr, tc] = int(move.promotion) * sign

            # Castling: move the rook
            if pt == PieceType.KING:
                dc = tc - fc
                if dc == 2:    # king-side
                    rook_c = 7
                    dest_c = 5
                elif dc == -2: # queen-side
                    rook_c = 0
                    dest_c = 3
                else:
                    rook_c = None
                if rook_c is not None:
                    self.board[fr, dest_c] = self.board[fr, rook_c]
                    self.board[fr, rook_c] = 0

                # Revoke castling rights for this color
                if player == Color.WHITE:
                    self.castling_rights[0] = False
                    self.castling_rights[1] = False
                else:
                    self.castling_rights[2] = False
                    self.castling_rights[3] = False

            # Revoke castling right if a rook moved from its home square
            if pt == PieceType.ROOK:
                if (fr, fc) == (7, 0): self.castling_rights[1] = False
                elif (fr, fc) == (7, 7): self.castling_rights[0] = False
                elif (fr, fc) == (0, 0): self.castling_rights[3] = False
                elif (fr, fc) == (0, 7): self.castling_rights[2] = False

        # Halfmove clock
        self.halfmove_clock = 0 if is_pawn_or_capture else self.halfmove_clock + 1

        # Switch turn
        if player == Color.BLACK:
            self.fullmove_number += 1
        self.current_player = opp

        self.position_history.append(self._position_key())

    def apply_move(self, move: Move) -> None:
        """Apply a (assumed legal) move to the state."""
        self._apply_unchecked(move)

    # ------------------------------------------------------------------
    # Terminal / result
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        if not self.get_legal_moves():
            return True
        if self.halfmove_clock >= 100:
            return True
        key = self._position_key()
        if self.position_history.count(key) >= 3:
            return True
        return False

    def get_result(self) -> float:
        """
        Result from the CURRENT player's perspective:
          +1 = current player wins,  -1 = current player loses,  0 = draw.
        Call only when is_terminal() is True.
        """
        legal = self.get_legal_moves()
        if not legal:
            # Checkmate → previous mover wins → current player loses
            if self.is_in_check(self.current_player):
                return -1.0
            return 0.0   # stalemate
        return 0.0        # 50-move or threefold repetition → draw

    def get_winner(self) -> Optional[Color]:
        """Returns the winner Color, or None if draw / game not over."""
        if not self.is_terminal():
            return None
        if self.is_in_check(self.current_player) and not self.get_legal_moves():
            return self.current_player.opponent()
        return None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_checkmate(self) -> bool:
        return (self.is_in_check(self.current_player)
                and not self.get_legal_moves())

    def is_stalemate(self) -> bool:
        return (not self.is_in_check(self.current_player)
                and not self.get_legal_moves())

    def __repr__(self) -> str:
        rows = []
        for r in range(8):
            row_str = []
            for c in range(8):
                p = int(self.board[r, c])
                if p == 0:
                    row_str.append(".")
                else:
                    letters = {1:"P",2:"R",3:"N",4:"B",5:"Q",6:"K"}
                    ch = letters[abs(p)]
                    row_str.append(ch if p > 0 else ch.lower())
            rows.append(" ".join(row_str))
        return "\n".join(rows)
