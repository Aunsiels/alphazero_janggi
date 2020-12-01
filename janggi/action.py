import torch

from janggi.utils import BOARD_HEIGHT, BOARD_WIDTH, get_symmetries, DEVICE

SYMMETRY_X = [9, 10, 11, 12, 13, 14, 15, 16, 17,  # North
              0, 1, 2, 3, 4, 5, 6, 7, 8,  # South
              18, 19, 20, 21, 22, 23, 24, 25,  # East
              26, 27, 28, 29, 30, 31, 32, 33,  # West
              35, 34, 37, 36, 39, 38, 41, 40,  # Diagonal
              43, 42, 45, 44, 48, 49, 46, 47,  # Horse
              51, 50, 53, 52, 56, 57, 54, 55  # Elephant
              ]

SYMMETRY_Y = [0, 1, 2, 3, 4, 5, 6, 7, 8,  # North
              9, 10, 11, 12, 13, 14, 15, 16, 17,  # South
              26, 27, 28, 29, 30, 31, 32, 33,  # East
              18, 19, 20, 21, 22, 23, 24, 25,  # West
              36, 37, 34, 35, 40, 41, 38, 39,  # Diagonal
              44, 45, 42, 43, 47, 46, 49, 48,  # Horse
              52, 53, 50, 51, 55, 54, 57, 56  # Elephant
              ]


class Action(object):

    __slots__ = ('x_from', 'y_from', "x_to", "y_to", "eaten", "_hash")

    def __init__(self, x_from, y_from, x_to, y_to):
        self.x_from = x_from
        self.y_from = y_from
        self.x_to = x_to
        self.y_to = y_to
        self.eaten = None
        self._hash = hash((self.x_from, self.y_from, self.x_to, self.y_to))

    def get_x_from(self, symmetry=False):
        if symmetry:
            return BOARD_HEIGHT - 1 - self.x_from
        return self.x_from

    def get_x_to(self, symmetry=False):
        if symmetry:
            return BOARD_HEIGHT - 1 - self.x_to
        return self.x_from

    def get_y_from(self, symmetry=False):
        if symmetry:
            return BOARD_WIDTH - 1 - self.y_from
        return self.y_from

    def get_y_to(self, symmetry=False):
        if symmetry:
            return BOARD_WIDTH - 1 - self.y_to
        return self.y_to

    def __str__(self):
        return "Action(" + str(self.x_from) + ", " + str(self.y_from) + ", " \
               + str(self.x_to) + ", " + str(self.y_to) + ")"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.x_from == other.x_from and self.y_from == other.y_from \
            and self.x_to == other.x_to and self.y_to == other.y_to

    def get_features(self, symmetry_x=False, symmetry_y=False):
        res = None
        if self.y_from == self.y_to:
            # Vertical move
            if self.x_to > self.x_from:
                # North move
                res = self.x_to - self.x_from - 1
            else:
                # South move
                res = self.x_from - self.x_to + 8
        elif self.x_from == self.x_to:
            # Horizontal move
            if self.y_to > self.y_from:
                # East move
                res = self.y_to - self.y_from + 17

            else:
                # West move
                res = self.y_from - self.y_to + 25
        elif self.y_to - self.y_from == 1:
            # Short Diagonal East
            if self.x_to - self.x_from == 1:
                res = 34
            elif self.x_to - self.x_from == -1:
                res = 35
            # Horse Vertical East
            elif self.x_to > self.x_from:
                res = 42
            else:
                res = 43
        elif self.y_to - self.y_from == -1:
            # Short Diagonal West
            if self.x_to - self.x_from == 1:
                res = 36
            elif self.x_to - self.x_from == -1:
                res = 37
            # Horse Vertical West
            elif self.x_to > self.x_from:
                res = 44
            else:
                res = 45
        elif self.x_to - self.x_from == 1:
            # Horse Horizontal North
            if self.y_to > self.y_from:
                res = 46
            else:
                res = 47
        elif self.x_to - self.x_from == -1:
            # Horse Horizontal South
            if self.y_to > self.y_from:
                res = 48
            else:
                res = 49
        elif self.y_to - self.y_from == 2:
            # Large Diagonal East
            if self.x_to - self.x_from == 2:
                res = 38
            elif self.x_to - self.x_from == -2:
                res = 39
            # Elephant Vertical East
            elif self.x_to > self.x_from:
                res = 50
            else:
                res = 51
        elif self.y_to - self.y_from == -2:
            # Large Diagonal West
            if self.x_to - self.x_from == 2:
                res = 40
            elif self.x_to - self.x_from == -2:
                res = 41
            # Elephant Vertical West
            elif self.x_to > self.x_from:
                res = 52
            else:
                res = 53
        elif self.x_to - self.x_from == 2:
            # Elephant Horizontal North
            if self.y_to > self.y_from:
                res = 54
            else:
                res = 55
        elif self.x_to - self.x_from == -2:
            # Horse Horizontal South
            if self.y_to > self.y_from:
                res = 56
            else:
                res = 57
        if symmetry_x:
            res = SYMMETRY_X[res]
        if symmetry_y:
            res = SYMMETRY_Y[res]
        return res

    def get_policy(self, current_player, data_augmentation=False):
        policy = torch.zeros((58, 10, 9))
        symmetry_x, symmetry_y = get_symmetries(current_player, data_augmentation)
        policy[self.get_features(symmetry_x, symmetry_y),
               self.get_x_from(symmetry_x), self.get_y_from(symmetry_y)] = 1.0
        # return policy.to(DEVICE)
        return policy


def get_none_action_policy(current_player, data_augmentation=False):
    policy = torch.zeros((58, 10, 9))
    # return policy.to(DEVICE)
    return policy
