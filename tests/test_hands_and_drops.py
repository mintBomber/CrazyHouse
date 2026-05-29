import unittest

from game.board import GameState
from game.pieces import Color, Move, PieceType


class HandsAndDropsTest(unittest.TestCase):
    def _empty_state(self, current_player=Color.WHITE):
        state = GameState()
        state.board[:] = 0
        state.board[7, 4] = int(PieceType.KING)
        state.board[0, 4] = -int(PieceType.KING)
        state.current_player = current_player
        state.castling_rights = [False, False, False, False]
        state.en_passant = None
        state.halfmove_clock = 0
        state.fullmove_number = 1
        state.position_history = [state._position_key()]
        return state

    def test_capture_adds_piece_to_capturers_hand(self):
        state = self._empty_state(Color.WHITE)
        state.board[4, 4] = int(PieceType.ROOK)
        state.board[4, 6] = -int(PieceType.KNIGHT)

        move = Move((4, 4), (4, 6))
        self.assertIn(move, state.get_legal_moves())

        state.apply_move(move)

        self.assertEqual(state.hands[Color.WHITE][PieceType.KNIGHT], 1)
        self.assertEqual(state.hands[Color.BLACK][PieceType.KNIGHT], 0)
        self.assertEqual(state.board[4, 6], int(PieceType.ROOK))
        self.assertEqual(state.current_player, Color.BLACK)

    def test_white_drop_uses_white_piece_color(self):
        state = self._empty_state(Color.WHITE)
        state.hands[Color.WHITE][PieceType.KNIGHT] = 1

        move = Move(None, (3, 3), is_drop=True, drop_piece=PieceType.KNIGHT)
        self.assertIn(move, state.get_legal_moves())

        state.apply_move(move)

        self.assertEqual(state.board[3, 3], int(PieceType.KNIGHT))
        self.assertEqual(state.hands[Color.WHITE][PieceType.KNIGHT], 0)
        self.assertEqual(state.current_player, Color.BLACK)

    def test_black_drop_uses_black_piece_color(self):
        state = self._empty_state(Color.BLACK)
        state.hands[Color.BLACK][PieceType.BISHOP] = 1

        move = Move(None, (4, 4), is_drop=True, drop_piece=PieceType.BISHOP)
        self.assertIn(move, state.get_legal_moves())

        state.apply_move(move)

        self.assertEqual(state.board[4, 4], -int(PieceType.BISHOP))
        self.assertEqual(state.hands[Color.BLACK][PieceType.BISHOP], 0)
        self.assertEqual(state.current_player, Color.WHITE)

    def test_checking_drop_is_legal_when_it_is_not_mate(self):
        state = self._empty_state(Color.WHITE)
        state.hands[Color.WHITE][PieceType.ROOK] = 1

        move = Move(None, (2, 4), is_drop=True, drop_piece=PieceType.ROOK)
        self.assertIn(move, state.get_legal_moves())

        state.apply_move(move)

        self.assertTrue(state.is_in_check(Color.BLACK))
        self.assertEqual(state.current_player, Color.BLACK)

    def test_drop_checkmate_is_illegal_for_any_piece_type(self):
        state = self._empty_state(Color.WHITE)
        state.hands[Color.WHITE][PieceType.ROOK] = 1
        state.board[2, 4] = int(PieceType.KING)
        state.board[7, 4] = 0
        state.board[0, 3] = -int(PieceType.ROOK)
        state.board[0, 5] = -int(PieceType.ROOK)
        state.board[1, 3] = -int(PieceType.KNIGHT)
        state.board[1, 5] = -int(PieceType.KNIGHT)
        state.position_history = [state._position_key()]

        move = Move(None, (1, 4), is_drop=True, drop_piece=PieceType.ROOK)

        self.assertFalse(state.is_in_check(Color.WHITE))
        self.assertNotIn(move, state.get_legal_moves())


if __name__ == "__main__":
    unittest.main()
