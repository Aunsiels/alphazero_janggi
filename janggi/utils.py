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
