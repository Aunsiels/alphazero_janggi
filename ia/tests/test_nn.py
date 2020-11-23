import unittest

from ia.janggi_network import FirstLayerJanggiNetwork, ResidualBlock, JanggiNetwork, PolicyNetwork, ValueNetwork
from janggi.board import Board
from janggi.utils import Color, BOARD_HEIGHT, BOARD_WIDTH


class TestNN(unittest.TestCase):

    def test_board(self):
        board = Board()
        first_layer_janggi_nn = FirstLayerJanggiNetwork()
        features_in = board.get_features(Color.BLUE, 1)
        features_out = first_layer_janggi_nn(features_in)
        self.assertEqual(list(features_out.shape), [1, 256, 10, 9])

        residual_block = ResidualBlock()
        features_out2 = residual_block(features_out)
        self.assertEqual(list(features_out2.shape), [1, 256, 10, 9])

        policy_network = PolicyNetwork()
        features_out3 = policy_network(features_out2)
        self.assertEqual(list(features_out3.shape), [1, 58, 10, 9])

        value_network = ValueNetwork()
        features_out4 = value_network(features_out2)
        self.assertEqual(list(features_out4.shape), [1, 1])

    def test_complete(self):
        board = Board()
        janggi_nn = JanggiNetwork()
        features_in = board.get_features(Color.BLUE, 1)
        features_in = features_in.view(1, -1, BOARD_HEIGHT, BOARD_WIDTH)
        policy, value = janggi_nn(features_in)
        self.assertEqual(list(policy.shape), [1, 58, 10, 9])
        self.assertEqual(list(value.shape), [1, 1])


if __name__ == '__main__':
    unittest.main()
