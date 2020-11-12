import unittest

from ia.random_mcts_player import RandomMCTSPlayer
from janggi.board import Board
from janggi.game import Game
from janggi.player import RandomPlayer
from janggi.utils import Color


class TestIA(unittest.TestCase):

    def test_mcts_vs_random(self):
        board = Board()
        player_blue = RandomPlayer(Color.BLUE)
        player_red = RandomMCTSPlayer(Color.RED)
        game = Game(player_blue, player_red, board)
        winner = game.run_game(200)
        self.assertEqual(winner, Color.RED)


if __name__ == '__main__':
    unittest.main()