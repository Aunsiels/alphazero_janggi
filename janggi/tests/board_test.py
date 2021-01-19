import unittest

import numpy as np

from janggi.action import Action
from janggi.board import Board, ActionCacheNode
from janggi.game import Game
from janggi.player import RandomPlayer
from janggi.utils import BOARD_HEIGHT, BOARD_WIDTH, Color
from janggi.piece import Soldier, Cannon, General, Chariot, Elephant, Horse, Guard


class BoardTest(unittest.TestCase):

    def setUp(self) -> None:
        self.board = Board()

    def test_number_pieces(self):
        count_blue = 0
        count_red = 0
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                if self.board.get(x, y) is not None:
                    if self.board.get(x, y).color == Color.BLUE:
                        count_blue += 1
                    else:
                        count_red += 1
        self.assertEqual(count_blue, 16)
        self.assertEqual(count_red, 16)

    def test_matching_placement(self):
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                if self.board.get(x, y) is not None:
                    self.assertEqual(x, self.board.get(x, y).x)
                    self.assertEqual(y, self.board.get(x, y).y)

    def test_color_right_side(self):
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                if self.board.get(x, y) is not None:
                    if self.board.get(x, y).color == Color.BLUE:
                        self.assertLess(x, 4)
                    else:
                        self.assertGreater(x, 5)

    def test_action_soldier(self):
        self.assertEqual(len(self.board.get(3, 0).get_actions()), 2)
        self.assertEqual(len(self.board.get(3, 2).get_actions()), 3)
        self.assertEqual(len(self.board.get(3, 8).get_actions()), 2)
        self.assertTrue(any([action.x_to == 4 for action in self.board.get(3, 0).get_actions()]))
        self.assertTrue(any([action.x_to == 5 for action in self.board.get(6, 0).get_actions()]))
        self.assertTrue(any([action.y_to == 1 for action in self.board.get(3, 0).get_actions()]))
        self.assertTrue(any([action.y_to == 1 for action in self.board.get(6, 0).get_actions()]))
        self.board.invalidate_action_cache()
        self.board.set(7, 3, Soldier(7, 3, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(7, 3).get_actions()), 4)

    def test_action_cannon(self):
        self.assertEqual(len(self.board.get(2, 1).get_actions()), 0)
        self.board.set(2, 0, Cannon(2, 0, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(2, 0).get_actions()), 3)
        self.board.set(4, 0, Soldier(4, 0, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(2, 0).get_actions()), 0)
        self.board.set(7, 3, Cannon(7, 3, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(7, 3).get_actions()), 1)
        self.board.set(2, 3, Cannon(2, 3, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(2, 3).get_actions()), 0)
        self.board.set(0, 5, None)
        self.assertEqual(len(self.board.get(2, 3).get_actions()), 1)
        self.board.set(2, 3, None)
        self.board.set(0, 5, Cannon(0, 5, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(0, 5).get_actions()), 1)

    def test_action_general(self):
        self.assertEqual(len(self.board.get(1, 4).get_actions()), 6)
        self.assertEqual(len(self.board.get(8, 4).get_actions()), 6)
        self.board.set(1, 4, None)
        self.board.set(2, 3, General(2, 3, Color.BLUE, self.board))
        self.board._current_action_cache_node = ActionCacheNode(None)
        self.assertEqual(len(self.board.get(2, 3).get_actions()), 3)

    def test_action_chariot(self):
        self.assertEqual(len(self.board.get(0, 0).get_actions()), 2)
        self.assertEqual(len(self.board.get(0, 8).get_actions()), 2)
        self.assertEqual(len(self.board.get(9, 0).get_actions()), 2)
        self.assertEqual(len(self.board.get(9, 8).get_actions()), 2)
        self.board.set(4, 0, Chariot(4, 0, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(4, 0).get_actions()), 10)
        self.board.set(7, 3, Chariot(7, 3, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(7, 3).get_actions()), 15)
        self.board.set(8, 4, None)
        self.assertEqual(len(self.board.get(7, 3).get_actions()), 16)

    def test_diagonal_chariot(self):
        self.board.set(1, 4, Chariot(1, 4, Color.BLUE, self.board))
        self.board.set(0, 3, None)
        self.board.set(0, 5, None)
        self.board._initialise_pieces_per_color()
        actions = self.board.get(1, 4).get_actions()
        self.assertEqual(len(actions), 14)

    def test_action_elephant(self):
        self.assertEqual(len(self.board.get(0, 1).get_actions()), 1)
        self.assertEqual(len(self.board.get(0, 7).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 1).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 7).get_actions()), 1)

        self.board.set(0, 1, None)
        self.board.set(3, 3, Elephant(3, 3, Color.BLUE, self.board))
        self.assertEqual(len(self.board.get(3, 3).get_actions()), 3)

    def test_action_horse(self):
        self.assertEqual(len(self.board.get(0, 2).get_actions()), 1)
        self.assertEqual(len(self.board.get(0, 6).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 2).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 6).get_actions()), 1)

    def test_action_guards(self):
        self.assertEqual(len(self.board.get(0, 3).get_actions()), 2)
        self.board.set(0, 3, None)
        self.board.set(1, 3, Guard(1, 3, Color.BLUE, self.board))
        self.board.invalidate_action_cache()
        self.board._initialise_pieces_per_color()
        self.assertEqual(len(self.board.get(1, 3).get_actions()), 2)

    def test_get_all_actions(self):
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 31)
        self.board.set(2, 3, Chariot(2, 3, Color.RED, self.board))
        self.board._current_action_cache_node = ActionCacheNode(None)
        self.board._initialise_pieces_per_color()
        actions = self.board.get_actions(Color.BLUE)
        self.assertEqual(len(actions), 4)

    def test_score(self):
        self.assertEqual(self.board.get_score(Color.BLUE), 72)
        self.assertEqual(self.board.get_score(Color.RED), 73.5)

    def test_get(self):
        self.assertEqual(self.board.get(0, 0).x, 0)
        self.assertEqual(self.board.get(0, 0).y, 0)
        self.assertEqual(self.board.get(BOARD_HEIGHT - 1, BOARD_WIDTH - 1, True).x, 0)
        self.assertEqual(self.board.get(BOARD_HEIGHT - 1, BOARD_WIDTH - 1, True).y, 0)

    def test_features(self):
        features = self.board.get_features(Color.BLUE, 10)
        self.assertEqual(features.shape, (16, 10, 9))
        self.assertEqual(features[0, 3, 0], 1)
        self.assertEqual(features[7, 6, 0], 1)
        self.assertEqual(features[-1, 6, 0], 10)
        features_red = self.board.get_features(Color.RED, 10)
        if self.board.start_red == self.board.start_blue:
            self.assertEqual(features[:14, :, :].tolist(), features_red[:14, :, :].tolist())

    def test_action_features(self, no_sum=False):
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                self.board.set(x, y, None)
        features = np.zeros(58)
        features_sym_y = np.zeros(58)
        features_sym_x = np.zeros(58)
        features_sym_x_y = np.zeros(58)
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                for piece in [Soldier, Cannon, General, Chariot, Elephant, Horse, Guard]:
                    for color in [Color.RED, Color.BLUE]:
                        self.board.set(x, y, piece(x, y, color, self.board))
                        for action in self.board.get(x, y).get_actions():
                            if not no_sum:
                                f_temp = action.get_features()
                                features[f_temp] += 1
                                f_temp = action.get_features(symmetry_y=True)
                                features_sym_y[f_temp] += 1
                                f_temp = action.get_features(symmetry_x=True)
                                features_sym_x[f_temp] += 1
                                f_temp = action.get_features(symmetry_x=True, symmetry_y=True)
                                features_sym_x_y[f_temp] += 1
                        self.board.set(x, y, None)
                        self._current_action_cache_node = ActionCacheNode(None)
        if not no_sum:
            self.assertTrue(not any([x == 0 for x in features]))
            self.assertTrue(not any([x == 0 for x in features_sym_y]))
            self.assertTrue(not any([x == 0 for x in features_sym_x]))
            self.assertTrue(not any([x == 0 for x in features_sym_x_y]))
            self.assertTrue(all(x == 0 for x in (features - features_sym_y)))
            self.assertTrue(all(x == 0 for x in (features - features_sym_x)))
            self.assertTrue(all(x == 0 for x in (features - features_sym_x_y)))

    def _test_perf(self):
        for _ in range(1000):
            self.test_action_features(no_sum=True)

    def perf_for_one_piece(self, piece):
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                if self.board.get(x, y) is not None:
                    continue
                for color in [Color.RED, Color.BLUE]:
                    self.board.set(x, y, piece(x, y, color, self.board))
                    self.board.get(x, y).get_actions()
                    self.board.set(x, y, None)
                    self._current_action_cache_node = ActionCacheNode(None)

    def _test_perf_chariot(self):
        for _ in range(1000):
            self.perf_for_one_piece(Chariot)

    def _test_perf_cannon(self):
        for _ in range(10000):
            self.perf_for_one_piece(Cannon)

    def test_read(self):
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                self.board.set(x, y, None)
        self.board.set(0, 5, Guard(0, 5, Color.BLUE, self.board))
        self.board.set(0, 7, Elephant(0, 7, Color.BLUE, self.board))
        self.board.set(1, 4, Guard(1, 4, Color.BLUE, self.board))
        self.board.set(2, 0, Horse(2, 0, Color.BLUE, self.board))
        self.board.set(2, 3, General(2, 3, Color.BLUE, self.board))
        self.board.set(4, 6, Cannon(4, 6, Color.BLUE, self.board))
        self.board.set(5, 0, Soldier(5, 0, Color.BLUE, self.board))
        self.board.set(7, 0, Chariot(0, 5, Color.BLUE, self.board))

        self.board.set(1, 8, Cannon(1, 8, Color.RED, self.board))
        self.board.set(3, 3, Soldier(3, 3, Color.RED, self.board))
        self.board.set(3, 5, Soldier(3, 5, Color.RED, self.board))
        self.board.set(3, 8, Soldier(3, 8, Color.RED, self.board))
        self.board.set(6, 8, Chariot(6, 8, Color.RED, self.board))
        self.board.set(7, 4, Guard(7, 4, Color.RED, self.board))
        self.board.set(8, 4, Horse(8, 4, Color.RED, self.board))
        self.board.set(9, 2, Horse(9, 2, Color.RED, self.board))
        self.board.set(9, 4, General(9, 4, Color.RED, self.board))
        self.board.set(9, 6, Guard(9, 6, Color.RED, self.board))
        self.board.set(9, 7, Elephant(9, 7, Color.RED, self.board))

        self.board._initialise_pieces_per_color()
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 1)

    def test_read_strange(self):
        board = Board.from_fen("1bnaa1bn1/R8/5k1cr/1p2p1B1p/2p6/9/1PP2P2P/4CCN2/1N2K4/2BA1A2R w - - 0 1")
        self.assertNotEqual(board.get_actions(Color.RED), [Action(0, 0, 0, 0)])


class TestBoardWon(BoardTest):

    def setUp(self) -> None:
        self.board = Board(start_blue="won", start_red="won")

    def test_action_elephant(self):
        self.assertEqual(len(self.board.get(0, 2).get_actions()), 0)
        self.assertEqual(len(self.board.get(0, 6).get_actions()), 0)
        self.assertEqual(len(self.board.get(9, 2).get_actions()), 0)
        self.assertEqual(len(self.board.get(9, 6).get_actions()), 0)
        self.board.set(0, 1, None)
        self.board.set(3, 3, Elephant(3, 3, Color.BLUE, self.board))
        self.board.invalidate_action_cache()
        self.assertEqual(len(self.board.get(3, 3).get_actions()), 3)

    def test_action_horse(self):
        self.assertEqual(len(self.board.get(0, 1).get_actions()), 2)
        self.assertEqual(len(self.board.get(0, 7).get_actions()), 2)
        self.assertEqual(len(self.board.get(9, 1).get_actions()), 2)
        self.assertEqual(len(self.board.get(9, 7).get_actions()), 2)

    def test_get_all_actions(self):
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 31)


class TestBoardGwee(BoardTest):

    def setUp(self) -> None:
        self.board = Board(start_blue="gwee", start_red="gwee")

    def test_action_elephant(self):
        self.assertEqual(len(self.board.get(0, 1).get_actions()), 1)
        self.assertEqual(len(self.board.get(0, 6).get_actions()), 0)
        self.assertEqual(len(self.board.get(9, 2).get_actions()), 0)
        self.assertEqual(len(self.board.get(9, 7).get_actions()), 1)
        self.board.set(0, 1, None)
        self.board.set(3, 3, Elephant(3, 3, Color.BLUE, self.board))
        self.board.invalidate_action_cache()
        self.assertEqual(len(self.board.get(3, 3).get_actions()), 3)

    def test_action_horse(self):
        self.assertEqual(len(self.board.get(0, 2).get_actions()), 1)
        self.assertEqual(len(self.board.get(0, 7).get_actions()), 2)
        self.assertEqual(len(self.board.get(9, 6).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 1).get_actions()), 2)

    def test_get_all_actions(self):
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 31)

    def test_read(self):
        board = Board.from_string("""
            .HEG.GE.R
            .....K...
            .......RH
            S.S...C.S
            .......S.
            .s..sr.C.
            s......ss
            ...kce.c.
            .........
            .h..ggher""")
        self.assertEqual(len(board.get_actions(Color.BLUE)), 1)
        self.assertFalse(board.is_finished(Color.BLUE))

    def test_read2(self):
        board = Board.from_string("""
            ...K.....
            cH.......
            .H..G....
            ......s..
            ...S..s..
            ....s....
            .........
            .hRk.gh..
            .C..gE...
            .........""")
        self.assertEqual(len(board.get_actions(Color.RED)), 1)
        self.assertFalse(board.is_finished(Color.RED))

    def test_to_fen(self):
        fen = self.board.to_fen(Color.BLUE, 105)
        self.assertIn("/", fen)
        self.assertIn("54", fen)

    def test_from_fen(self):
        board = Board.from_fen("rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1")
        self.assertEqual(board.get_score(Color.BLUE), 72)

    def test_game_from_fen(self):
        game = Game.from_fen(RandomPlayer(Color.BLUE),
                             RandomPlayer(Color.RED),
                             "rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1")
        self.assertEqual(game.current_player, Color.BLUE)
        self.assertEqual(game.round, 0)

    def test_uci_usi(self):
        game = Game.from_uci_usi(
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.RED),
            "position fen r1bakab1r/9/2ncc4/1pppn2R1/9/9/BPP1P1P1P/2N1CN1C1/4K4/3A1A2R w - - 0 1 "
            "moves g4h4 f10e9 i1g1 d10d9 g1g9 d8d10 h3h5 a10a6 e3i3 e7f5 h5e5 a6e6 i3i10 d10f10 "
            "h7f7 f5h4 e5e3 e6i6 g9g10 h4f3 g10f10")
        self.assertIsNotNone(game)

    def test_uci_usi4(self):
        game = Game.from_uci_usi(
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.RED),
            "position fen rbna1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1 "
            "moves b1c3 a7b7 b3e3 a10a4 c3a4 f10e10 e4d4 e7f7 g1e4 e9f9 h1g3 c7c6 h3f3 f7e7 e4g7 b8b2 a4b2 i10i8 a0a9 "
            "f9f8 g4f4")
        action = game.get_current_actions()[0]
        game.apply_action(action, invalidate_cache=False)
        game.reverse_action()
        print(game.get_current_actions())

    def test_uci_usi3(self):
        uci_usi = "position fen rbna1anbr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RBNA1ABNR w - - 0 1 moves i4h4 " \
                  "g10f8 h1g3 b8g8 a4b4 g7f7 g4f4 g8g2 c1d3 e9e9 b3e3 h8e8 h4g4 g2c2 b1d4 b10d7 h3f3 i10i8 i1h1 d7f4 " \
                  "g4f4 i8g8 g3i4 i7h7 h1h7 g8g2 e2e1 f8h7 d1e2 c10d8 f3d1 d10e10 f1f2 c7b7 a1a5 c2f2 e2f2 e7e6 f2e2 " \
                  "f7f6 a5h5 g2g7 h5h6 a10d10 d1f1 e9f9 f4f5 f9e9 f5f6 e6d6 f6g6 g7d7 e4e5 e9d9 i4g5 e8b8 g6f6 d7g7 " \
                  "g5h7 g7g1 h7i9 g1i1 i9g8 b8g8 h6h10 f10e9 e2f2 g8b8 e1e2 b8e8 d3e1 i1i3 f2f3 b7c7 f1d1 c7d7 d4g6 " \
                  "i3i7 f6e6 d8b7 e6d6 i7i2 f3f2 b7d6 e5e6 i2i5 e6d6 d10b10 d6d7 i5d5 g6d4 d5d7 d1d7 b10a10 d7d1 e9d8 " \
                  "h10h9 d9d10 d1d8 e10f10 h9h10 d10e10 d8d1 a7b7 d1f1 a10a2 e2d3 a2f2 f1d1 b7c7 h10h8 e10e9 d4g6 " \
                  "f2f8 h8h9 e9e10 h9c9 f10e9 c9c10 e9d10 c10c7 e10f10 c7e7 d10e9 e7i7 f10f9 i7i9 f9f10 e1g2 e8e10 " \
                  "g2h4 e9f9 h4i6 f8f4 i6h8 f4f1 d3d2 f9f8 g6d4 f1f2 d2d3 f2f3 i9i1 f8e9 i1e1 f3f4 h8g6 f4f6 d3d3 " \
                  "e10g10 d3d3 f6f2 e1e2 f2e2 d3e2 g10e10 e2e2 e10e4 e2e2 f10e10 g6e7 e4e8 e7d5 e10f10 d5c7 e8e10 " \
                  "d1d6 e10e7 d6d3 e7e10 d3f1 e10e5 e2f2 f10e10 d4g2 e9d9 g2e5 d9e9 e5b7 e9f8 c7e8 f8e9 e8g9 e9f9 " \
                  "f1f9 e10f10 f9f1 f10f10"
        game = Game.from_uci_usi(
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.RED),
            uci_usi)
        self.assertIsNotNone(game)

    def test_strange_move_chariot(self):
        board = Board.from_fen("2b1akb1B/2r6/4C2c1/5p3/1p1P1Pp2/9/1PnN5/5R3/2B1K4/5AN2 w - - 1 67")
        actions = board.get(2, 5).get_actions()
        self.assertNotIn(Action(2, 5, 1, 4), actions)


class TestBoardSang(BoardTest):

    def setUp(self) -> None:
        self.board = Board(start_blue="sang", start_red="sang")

    def test_action_elephant(self):
        self.assertEqual(len(self.board.get(0, 2).get_actions()), 0)
        self.assertEqual(len(self.board.get(0, 7).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 1).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 6).get_actions()), 0)
        self.board.set(0, 1, None)
        self.board.set(3, 3, Elephant(3, 3, Color.BLUE, self.board))
        self.board.invalidate_action_cache()
        self.assertEqual(len(self.board.get(3, 3).get_actions()), 3)

    def test_action_horse(self):
        self.assertEqual(len(self.board.get(0, 1).get_actions()), 2)
        self.assertEqual(len(self.board.get(0, 6).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 2).get_actions()), 1)
        self.assertEqual(len(self.board.get(9, 7).get_actions()), 2)

    def test_get_all_actions(self):
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 31)


if __name__ == '__main__':
    unittest.main()
