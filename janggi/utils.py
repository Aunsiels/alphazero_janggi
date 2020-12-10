import random
import subprocess
from enum import Enum
from time import sleep

import torch

STOCKFISH_LOCATION = 'D:/Downloads/fairy-stockfish-largeboard_x86-64.exe'

BOARD_HEIGHT = 10
BOARD_WIDTH = 9


class Color(Enum):
    BLUE = 1
    RED = -1


if torch.cuda.is_available():
    DEV = "cuda:0"
else:
    DEV = "cpu"
DEVICE = torch.device(DEV)
CPU_DEVICE = torch.device("cpu")


def get_symmetries(current_player, data_augmentation=False):
    symmetry_x = False
    symmetry_y = False
    if current_player == Color.RED:
        symmetry_x = True
        symmetry_y = True
    if data_augmentation:
        symmetry_y = not symmetry_y
    return symmetry_x, symmetry_y


def get_random_board():
    from janggi.board import Board
    start_blue = random.choice(["won", "sang", "yang", "gwee"])
    start_red = random.choice(["won", "sang", "yang", "gwee"])
    board = Board(start_blue=start_blue, start_red=start_red)
    return board


def get_process_stockfish(board):
    process = subprocess.Popen([STOCKFISH_LOCATION],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               universal_newlines=True)
    sleep(0.1)
    process.stdin.write("xboard\n")
    process.stdin.write("variant janggi\n")
    process.stdin.write("setboard " + board.to_fen(Color.BLUE, 0) + "\n")
    process.stdin.flush()
    return process