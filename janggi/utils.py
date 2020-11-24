from enum import Enum

import torch

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