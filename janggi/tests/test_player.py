import unittest

from janggi.board import Board
from janggi.board_fish import BoardFish
from janggi.game import Game
from janggi.player import RandomPlayer
from janggi.utils import Color


class TestPlayer(unittest.TestCase):

    def test_random(self):
        player_blue = RandomPlayer(Color.BLUE)
        player_red = RandomPlayer(Color.RED)
        board = BoardFish()
        game = Game(player_blue, player_red, board)
        game.run_game(200)
        print(repr(board))
        print(board)
        print(game.get_winner())


if __name__ == '__main__':
    unittest.main()
