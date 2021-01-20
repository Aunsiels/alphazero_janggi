import unittest

from ia.janggi_network import JanggiNetwork
from ia.mcts import MCTSNode
from ia.random_mcts_player import RandomMCTSPlayer, fight, NNPlayer
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
        n_simulations = 400
        node = MCTSNode()
        player_blue = RandomMCTSPlayer(Color.BLUE, n_simulations=n_simulations, current_node=node)
        player_red = RandomMCTSPlayer(Color.RED, n_simulations=n_simulations, current_node=node)
        # winner = fight(player_blue, player_red, 200)
        board = get_random_board()
        game = Game(player_blue, player_red, board)
        winner = game.run_game(200)
        self.assertIn(winner, [Color.BLUE, Color.RED])
        print(game.to_json(node))

    def test_single_action_random(self):
        n_simulations = 16000
        node = MCTSNode()
        player_blue = RandomMCTSPlayer(Color.BLUE, n_simulations=n_simulations, current_node=node)
        player_red = RandomMCTSPlayer(Color.RED, n_simulations=n_simulations, current_node=node)
        board = get_random_board()
        game = Game(player_blue, player_red, board)
        game.get_next_action()

    def test_single_action_nn(self):
        n_simulations = 800
        player_blue = NNPlayer(Color.BLUE, n_simulations=n_simulations,
                               janggi_net=JanggiNetwork(),
                               temperature_start=0.01,
                               temperature_threshold=30,
                               temperature_end=0.01)
        player_red = NNPlayer(Color.RED, n_simulations=n_simulations,
                              janggi_net=JanggiNetwork(),
                              temperature_start=0.01,
                              temperature_threshold=30,
                              temperature_end=0.01)
        board = get_random_board()
        game = Game(player_blue, player_red, board)
        game.get_next_action()


if __name__ == '__main__':
    unittest.main()
