"""Tests for hexzero/game.py — HexState mechanics."""
import unittest

import numpy as np

from hexzero.game import BLACK, EMPTY, SWAP_MOVE, WHITE, HexState


class LegalMovesTests(unittest.TestCase):
    def test_empty_board_all_cells_legal(self):
        s = HexState(3)
        moves = s.legal_moves()
        self.assertEqual(len(moves), 9)
        self.assertNotIn(SWAP_MOVE, moves)

    def test_no_legal_moves_after_game_ends(self):
        # Fill a 2×2 board so BLACK wins on move 2
        s = HexState(2)
        s.apply_move((0, 0))  # BLACK top row
        s.apply_move((0, 1))  # WHITE
        s.apply_move((1, 0))  # BLACK bottom row → BLACK connects
        self.assertEqual(s.winner(), BLACK)
        self.assertEqual(s.legal_moves(), [])

    def test_swap_legal_only_on_move_1(self):
        s = HexState(3, pie_rule=True)
        self.assertNotIn(SWAP_MOVE, s.legal_moves())  # move 0
        s.apply_move((1, 1))
        self.assertIn(SWAP_MOVE, s.legal_moves())     # move 1
        s.apply_move((0, 0))                          # normal move (not swap)
        self.assertNotIn(SWAP_MOVE, s.legal_moves())  # move 2

    def test_swap_not_available_without_pie_rule(self):
        s = HexState(3, pie_rule=False)
        s.apply_move((1, 1))
        self.assertNotIn(SWAP_MOVE, s.legal_moves())

    def test_played_cells_not_in_legal_moves(self):
        s = HexState(3)
        s.apply_move((0, 0))
        moves = s.legal_moves()
        self.assertNotIn((0, 0), moves)
        self.assertEqual(len(moves), 8)


class ApplyMoveTests(unittest.TestCase):
    def test_stone_placed_correctly(self):
        s = HexState(3)
        s.apply_move((1, 2))
        self.assertEqual(s.board[1, 2], BLACK)

    def test_player_alternates(self):
        s = HexState(3)
        self.assertEqual(s.current_player, BLACK)
        s.apply_move((0, 0))
        self.assertEqual(s.current_player, WHITE)
        s.apply_move((0, 1))
        self.assertEqual(s.current_player, BLACK)

    def test_move_count_increments(self):
        s = HexState(3)
        self.assertEqual(s.move_count, 0)
        s.apply_move((0, 0))
        self.assertEqual(s.move_count, 1)
        s.apply_move((0, 1))
        self.assertEqual(s.move_count, 2)

    def test_last_move_updated(self):
        s = HexState(3)
        s.apply_move((2, 1))
        self.assertEqual(s.last_move, (2, 1))

    def test_swap_move_stone_stays_black(self):
        # Swapper adopts the BLACK role — the stone stays BLACK on the board.
        s = HexState(3, pie_rule=True)
        s.apply_move((1, 1))
        s.apply_move(SWAP_MOVE)
        self.assertEqual(s.board[1, 1], BLACK)
        self.assertEqual(int((s.board == BLACK).sum()), 1)
        self.assertEqual(int((s.board == WHITE).sum()), 0)

    def test_swap_move_gives_turn_to_white(self):
        # Original first player (now WHITE, left→right) plays next.
        s = HexState(3, pie_rule=True)
        s.apply_move((1, 1))
        s.apply_move(SWAP_MOVE)
        self.assertEqual(s.current_player, WHITE)


class WinDetectionTests(unittest.TestCase):
    def test_black_wins_top_to_bottom_2x2(self):
        s = HexState(2)
        s.apply_move((0, 0))  # BLACK
        s.apply_move((0, 1))  # WHITE
        s.apply_move((1, 0))  # BLACK — connects row 0 and row 1
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.winner(), BLACK)

    def test_white_wins_left_to_right_2x2(self):
        # WHITE needs col 0 to col 1
        s = HexState(2)
        s.apply_move((0, 0))  # BLACK
        s.apply_move((0, 1))  # WHITE col 1
        s.apply_move((1, 1))  # BLACK
        s.apply_move((1, 0))  # WHITE col 0 → WHITE connects
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.winner(), WHITE)

    def test_not_terminal_on_empty_board(self):
        s = HexState(3)
        self.assertFalse(s.is_terminal())
        self.assertIsNone(s.winner())

    def test_result_for_winner(self):
        s = HexState(2)
        s.apply_move((0, 0))
        s.apply_move((0, 1))
        s.apply_move((1, 0))  # BLACK wins
        self.assertEqual(s.result_for(BLACK), 1.0)
        self.assertEqual(s.result_for(WHITE), -1.0)

    def test_result_for_non_terminal(self):
        s = HexState(3)
        self.assertEqual(s.result_for(BLACK), 0.0)
        self.assertEqual(s.result_for(WHITE), 0.0)

    def test_3x3_black_column_win(self):
        # BLACK wins via left column: (0,0)→(1,0)→(2,0), each adjacent by delta (1,0)
        s = HexState(3)
        s.apply_move((0, 0))  # BLACK row 0
        s.apply_move((0, 1))  # WHITE
        s.apply_move((1, 0))  # BLACK row 1
        s.apply_move((0, 2))  # WHITE
        s.apply_move((2, 0))  # BLACK row 2 → BLACK wins top-to-bottom
        self.assertEqual(s.winner(), BLACK)


class WinningPathTests(unittest.TestCase):
    def test_empty_path_if_not_terminal(self):
        s = HexState(3)
        self.assertEqual(s.winning_path(), set())

    def test_winning_path_contains_winning_stones(self):
        s = HexState(2)
        s.apply_move((0, 0))  # BLACK
        s.apply_move((0, 1))  # WHITE
        s.apply_move((1, 0))  # BLACK wins
        path = s.winning_path()
        self.assertIn((0, 0), path)
        self.assertIn((1, 0), path)

    def test_winning_path_excludes_non_path_stones(self):
        # 3×3: BLACK wins via col 0 top-to-bottom; (0,2) is irrelevant
        s = HexState(3)
        s.apply_move((0, 0))  # BLACK
        s.apply_move((2, 2))  # WHITE
        s.apply_move((1, 0))  # BLACK
        s.apply_move((2, 1))  # WHITE
        s.apply_move((0, 2))  # BLACK — NOT on the winning path
        s.apply_move((0, 1))  # WHITE
        s.apply_move((2, 0))  # BLACK row 2 col 0 → win through col 0
        path = s.winning_path()
        self.assertIn((0, 0), path)
        self.assertIn((1, 0), path)
        self.assertIn((2, 0), path)


class CloneTests(unittest.TestCase):
    def test_clone_is_independent(self):
        s = HexState(3)
        s.apply_move((0, 0))
        c = s.clone()
        c.apply_move((1, 1))
        # Original should not see (1,1)
        self.assertEqual(s.board[1, 1], EMPTY)
        self.assertEqual(s.move_count, 1)

    def test_clone_preserves_winner(self):
        s = HexState(2)
        s.apply_move((0, 0))
        s.apply_move((0, 1))
        s.apply_move((1, 0))  # BLACK wins
        c = s.clone()
        self.assertEqual(c.winner(), BLACK)
        self.assertTrue(c.is_terminal())

    def test_clone_win_detection_consistent(self):
        # Clone before win; check that win detection still works in clone
        s = HexState(2)
        s.apply_move((0, 0))
        s.apply_move((0, 1))
        c = s.clone()
        c.apply_move((1, 0))  # BLACK wins in clone
        self.assertEqual(c.winner(), BLACK)
        # Original still non-terminal
        self.assertFalse(s.is_terminal())


class SymmetryTests(unittest.TestCase):
    def test_180_rotation_preserves_player(self):
        s = HexState(3)
        s.apply_move((0, 0))
        s.apply_move((2, 2))
        sym = s.apply_symmetry()
        self.assertEqual(sym.current_player, s.current_player)

    def test_180_rotation_maps_stones_correctly(self):
        s = HexState(3)
        s.apply_move((0, 1))  # BLACK at (0,1) → should become (2,1)
        sym = s.apply_symmetry()
        self.assertEqual(sym.board[2, 1], BLACK)
        self.assertEqual(sym.board[0, 1], EMPTY)

    def test_180_rotation_preserves_win_status(self):
        s = HexState(2)
        s.apply_move((0, 0))
        s.apply_move((0, 1))
        s.apply_move((1, 0))  # BLACK wins
        sym = s.apply_symmetry()
        self.assertEqual(sym.winner(), BLACK)

    def test_double_rotation_is_identity(self):
        s = HexState(3)
        s.apply_move((0, 1))
        s.apply_move((1, 2))
        sym2 = s.apply_symmetry().apply_symmetry()
        np.testing.assert_array_equal(sym2.board, s.board)

    def test_symmetry_preserves_move_count(self):
        s = HexState(4)
        s.apply_move((0, 0))
        s.apply_move((1, 1))
        sym = s.apply_symmetry()
        self.assertEqual(sym.move_count, s.move_count)


class PieRuleWinConsistencyTests(unittest.TestCase):
    """After a SWAP_MOVE, win detection must work correctly."""

    def test_swap_then_black_wins(self):
        # WHITE swaps → becomes BLACK (top→bottom), inherits stone at (1,1).
        # Original BLACK becomes WHITE (left→right) and plays next.
        # The swapper (now BLACK) wins top-to-bottom via (0,1)-(1,1)-(2,1).
        s = HexState(3, pie_rule=True)
        s.apply_move((1, 1))       # BLACK places center
        s.apply_move(SWAP_MOVE)    # WHITE swaps; stone stays BLACK; WHITE plays next
        # current_player = WHITE (original BLACK, now left→right)
        s.apply_move((2, 2))       # WHITE plays
        # current_player = BLACK (swapper, top→bottom), has stone at (1,1)
        # (0,1)↔(1,1): delta (1,0) ∈ _NEIGHBOUR_DELTAS ✓
        # (1,1)↔(2,1): delta (1,0) ∈ _NEIGHBOUR_DELTAS ✓
        s.apply_move((0, 1))       # BLACK — row 0
        s.apply_move((0, 0))       # WHITE plays
        s.apply_move((2, 1))       # BLACK — row 2; (0,1)-(1,1)-(2,1) = BLACK wins
        self.assertEqual(s.winner(), BLACK)


if __name__ == "__main__":
    unittest.main()
