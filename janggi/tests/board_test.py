import unittest

import numpy as np

from janggi.board import Board, ActionCacheNode
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
                if self.board.board[x][y] is not None:
                    if self.board.board[x][y].color == Color.BLUE:
                        count_blue += 1
                    else:
                        count_red += 1
        self.assertEqual(count_blue, 16)
        self.assertEqual(count_red, 16)

    def test_matching_placement(self):
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                if self.board.board[x][y] is not None:
                    self.assertEqual(x, self.board.board[x][y].x)
                    self.assertEqual(y, self.board.board[x][y].y)

    def test_color_right_side(self):
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                if self.board.board[x][y] is not None:
                    if self.board.board[x][y].color == Color.BLUE:
                        self.assertLess(x, 4)
                    else:
                        self.assertGreater(x, 5)

    def test_action_soldier(self):
        self.assertEqual(len(self.board.board[3][0].get_actions()), 2)
        self.assertEqual(len(self.board.board[3][2].get_actions()), 3)
        self.assertEqual(len(self.board.board[3][8].get_actions()), 2)
        self.assertTrue(any([action.x_to == 4 for action in self.board.board[3][0].get_actions()]))
        self.assertTrue(any([action.x_to == 5 for action in self.board.board[6][0].get_actions()]))
        self.assertTrue(any([action.y_to == 1 for action in self.board.board[3][0].get_actions()]))
        self.assertTrue(any([action.y_to == 1 for action in self.board.board[6][0].get_actions()]))
        self.board.invalidate_action_cache()
        self.board.board[7][3] = Soldier(7, 3, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[7][3].get_actions()), 4)

    def test_action_cannon(self):
        self.assertEqual(len(self.board.board[2][1].get_actions()), 0)
        self.board.board[2][0] = Cannon(2, 0, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[2][0].get_actions()), 3)
        self.board.board[4][0] = Soldier(4, 0, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[2][0].get_actions()), 0)
        self.board.board[7][3] = Cannon(7, 3, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[7][3].get_actions()), 1)
        self.board.board[2][3] = Cannon(2, 3, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[2][3].get_actions()), 0)
        self.board.board[0][5] = None
        self.assertEqual(len(self.board.board[2][3].get_actions()), 1)
        self.board.board[2][3] = None
        self.board.board[0][5] = Cannon(0, 5, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[0][5].get_actions()), 1)

    def test_action_general(self):
        self.assertEqual(len(self.board.board[1][4].get_actions()), 6)
        self.assertEqual(len(self.board.board[8][4].get_actions()), 6)
        self.board.board[1][4] = None
        self.board.board[2][3] = General(2, 3, Color.BLUE, self.board)
        self.board._current_action_cache_node = ActionCacheNode(None)
        self.assertEqual(len(self.board.board[2][3].get_actions()), 3)

    def test_action_chariot(self):
        self.assertEqual(len(self.board.board[0][0].get_actions()), 2)
        self.assertEqual(len(self.board.board[0][8].get_actions()), 2)
        self.assertEqual(len(self.board.board[9][0].get_actions()), 2)
        self.assertEqual(len(self.board.board[9][8].get_actions()), 2)
        self.board.board[4][0] = Chariot(4, 0, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[4][0].get_actions()), 10)
        self.board.board[7][3] = Chariot(7, 3, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[7][3].get_actions()), 15)
        self.board.board[8][4] = None
        self.assertEqual(len(self.board.board[7][3].get_actions()), 16)

    def test_diagonal_chariot(self):
        self.board.board[1][4] = Chariot(1, 4, Color.BLUE, self.board)
        self.board.board[0][3] = None
        self.board.board[0][5] = None
        self.board._initialise_pieces_per_color()
        actions = self.board.board[1][4].get_actions()
        self.assertEqual(len(actions), 14)

    def test_action_elephant(self):
        self.assertEqual(len(self.board.board[0][1].get_actions()), 1)
        self.assertEqual(len(self.board.board[0][7].get_actions()), 1)
        self.assertEqual(len(self.board.board[9][1].get_actions()), 1)
        self.assertEqual(len(self.board.board[9][7].get_actions()), 1)

        self.board.board[0][1] = None
        self.board.board[3][3] = Elephant(3, 3, Color.BLUE, self.board)
        self.assertEqual(len(self.board.board[3][3].get_actions()), 3)

    def test_action_horse(self):
        self.assertEqual(len(self.board.board[0][2].get_actions()), 1)
        self.assertEqual(len(self.board.board[0][6].get_actions()), 1)
        self.assertEqual(len(self.board.board[9][2].get_actions()), 1)
        self.assertEqual(len(self.board.board[9][6].get_actions()), 1)

    def test_action_guards(self):
        self.assertEqual(len(self.board.board[0][3].get_actions()), 2)

    def test_get_all_actions(self):
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 31)
        self.board.board[2][3] = Chariot(2, 3, Color.RED, self.board)
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

    def test_action_features(self, no_sum=False):
        self.board.board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        features = np.zeros(58)
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                for piece in [Soldier, Cannon, General, Chariot, Elephant, Horse, Guard]:
                    self.board.board[x][y] = piece(x, y, Color.BLUE, self.board)
                    for action in self.board.board[x][y].get_actions():
                        if not no_sum:
                            f_temp = action.get_features()
                            features[f_temp] += 1
                    self.board.board[x][y] = None
                    self._current_action_cache_node = ActionCacheNode(None)
        if not no_sum:
            self.assertTrue(not any([x == 0 for x in features]))

    def _test_perf(self):
        for _ in range(1000):
            self.test_action_features(no_sum=True)


class TestBoardWon(BoardTest):

    def setUp(self) -> None:
        self.board = Board(start_blue="won", start_red="won")

    def test_action_elephant(self):
        self.assertEqual(len(self.board.board[0][2].get_actions()), 0)
        self.assertEqual(len(self.board.board[0][6].get_actions()), 0)
        self.assertEqual(len(self.board.board[9][2].get_actions()), 0)
        self.assertEqual(len(self.board.board[9][6].get_actions()), 0)
        self.board.board[0][1] = None
        self.board.board[3][3] = Elephant(3, 3, Color.BLUE, self.board)
        self.board.invalidate_action_cache()
        self.assertEqual(len(self.board.board[3][3].get_actions()), 3)

    def test_action_horse(self):
        self.assertEqual(len(self.board.board[0][1].get_actions()), 2)
        self.assertEqual(len(self.board.board[0][7].get_actions()), 2)
        self.assertEqual(len(self.board.board[9][1].get_actions()), 2)
        self.assertEqual(len(self.board.board[9][7].get_actions()), 2)

    def test_get_all_actions(self):
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 31)

class TestBoardGwee(BoardTest):

    def setUp(self) -> None:
        self.board = Board(start_blue="gwee", start_red="gwee")

    def test_action_elephant(self):
        self.assertEqual(len(self.board.board[0][1].get_actions()), 1)
        self.assertEqual(len(self.board.board[0][6].get_actions()), 0)
        self.assertEqual(len(self.board.board[9][1].get_actions()), 1)
        self.assertEqual(len(self.board.board[9][6].get_actions()), 0)
        self.board.board[0][1] = None
        self.board.board[3][3] = Elephant(3, 3, Color.BLUE, self.board)
        self.board.invalidate_action_cache()
        self.assertEqual(len(self.board.board[3][3].get_actions()), 3)

    def test_action_horse(self):
        self.assertEqual(len(self.board.board[0][2].get_actions()), 1)
        self.assertEqual(len(self.board.board[0][7].get_actions()), 2)
        self.assertEqual(len(self.board.board[9][2].get_actions()), 1)
        self.assertEqual(len(self.board.board[9][7].get_actions()), 2)

    def test_get_all_actions(self):
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 31)

class TestBoardSang(BoardTest):

    def setUp(self) -> None:
        self.board = Board(start_blue="sang", start_red="sang")

    def test_action_elephant(self):
        self.assertEqual(len(self.board.board[0][2].get_actions()), 0)
        self.assertEqual(len(self.board.board[0][7].get_actions()), 1)
        self.assertEqual(len(self.board.board[9][2].get_actions()), 0)
        self.assertEqual(len(self.board.board[9][7].get_actions()), 1)
        self.board.board[0][1] = None
        self.board.board[3][3] = Elephant(3, 3, Color.BLUE, self.board)
        self.board.invalidate_action_cache()
        self.assertEqual(len(self.board.board[3][3].get_actions()), 3)

    def test_action_horse(self):
        self.assertEqual(len(self.board.board[0][1].get_actions()), 2)
        self.assertEqual(len(self.board.board[0][6].get_actions()), 1)
        self.assertEqual(len(self.board.board[9][1].get_actions()), 2)
        self.assertEqual(len(self.board.board[9][6].get_actions()), 1)

    def test_get_all_actions(self):
        self.assertEqual(len(self.board.get_actions(Color.BLUE)), 31)


if __name__ == '__main__':
    unittest.main()
