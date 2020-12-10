import unittest

from ia.mcts import MCTSNode
from ia.random_mcts_player import RandomMCTSPlayer, fight
from janggi.game import Game
from janggi.player import RandomPlayer
from janggi.utils import Color, get_random_board


class TestIA(unittest.TestCase):

    def test_mcts_vs_random(self):
        player_blue = RandomPlayer(Color.BLUE)
        player_red = RandomMCTSPlayer(Color.RED, n_simulations=200, temperature_start=0.01, temperature_threshold=30, temperature_end=0.01)
        winner = fight(player_blue, player_red, 200)
        self.assertEqual(winner, Color.RED)

    def test_random_vs_random(self):
        n_simulations = 200
        node = MCTSNode()
        player_blue = RandomMCTSPlayer(Color.BLUE, n_simulations=n_simulations, current_node=node)
        player_red = RandomMCTSPlayer(Color.RED, n_simulations=n_simulations, current_node=node)
        # winner = fight(player_blue, player_red, 200)
        board = get_random_board()
        game = Game(player_blue, player_red, board)
        winner = game.run_game(200)
        self.assertIn(winner, [Color.BLUE, Color.RED])
        print(game.to_json(node))


if __name__ == '__main__':
    unittest.main()
