import unittest

from ia.random_mcts_player import RandomMCTSPlayer, fight
from janggi.board import Board
from janggi.game import Game
from janggi.player import RandomPlayer
from janggi.utils import Color


class TestIA(unittest.TestCase):

    def _test_mcts_vs_random(self):
        player_blue = RandomPlayer(Color.BLUE)
        player_red = RandomMCTSPlayer(Color.RED, n_simulations=800, temperature_start=0.01, temperature_threshold=30, temperature_end=0.01)
        winner = fight(player_blue, player_red, 200)
        self.assertEqual(winner, Color.RED)

    def test_random_vs_random(self):
        n_simulations = 200
        player_blue = RandomMCTSPlayer(Color.BLUE, n_simulations=n_simulations)
        player_red = RandomMCTSPlayer(Color.RED, n_simulations=n_simulations)
        winner = fight(player_blue, player_red, 200)
        self.assertIn(winner, [Color.BLUE, Color.RED])


if __name__ == '__main__':
    unittest.main()
