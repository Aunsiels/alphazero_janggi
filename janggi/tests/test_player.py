import unittest

from janggi.board import Board
from janggi.game import Game
from janggi.player import RandomPlayer
from janggi.stockfish_player import StockfishPlayer
from janggi.utils import Color, get_random_board, get_process_stockfish


class TestPlayer(unittest.TestCase):

    def test_random(self):
        player_blue = RandomPlayer(Color.BLUE)
        player_red = RandomPlayer(Color.RED)
        board = Board()
        game = Game(player_blue, player_red, board)
        game.run_game(200)
        print(repr(board))
        print(board)
        print(game.get_winner())

    def test_stockfish(self):
        board = get_random_board()
        process = get_process_stockfish(board)
        player_blue = StockfishPlayer(Color.BLUE, process, think_time=2)
        player_red = StockfishPlayer(Color.RED, process, think_time=2)
        game = Game(player_blue, player_red, board)
        winner = game.run_game(200)
        print(winner)
        print(repr(game.board))


if __name__ == '__main__':
    unittest.main()
