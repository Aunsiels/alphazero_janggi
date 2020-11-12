import numpy as np


class Action(object):

    __slots__ = ('x_from', 'y_from', "x_to", "y_to", "eaten")

    def __init__(self, x_from, y_from, x_to, y_to):
        self.x_from = x_from
        self.y_from = y_from
        self.x_to = x_to
        self.y_to = y_to
        self.eaten = None

    def __str__(self):
        return "Action(" + str(self.x_from) + ", " + str(self.y_from) + ", " \
               + str(self.x_to) + ", " + str(self.y_to) + ")"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.x_from, self.y_from, self.x_to, self.y_to))

    def __eq__(self, other):
        return self.x_from == other.x_from and self.y_from == other.y_from \
            and self.x_to == other.x_to and self.y_to == other.y_to

    def get_features(self):
        if self.y_from == self.y_to:
            # Vertical move
            if self.x_to > self.x_from:
                # North move
                return self.x_to - self.x_from - 1
            else:
                # South move
                return self.x_from - self.x_to + 8
        elif self.x_from == self.x_to:
            # Horizontal move
            if self.y_to > self.y_from:
                # East move
                return self.y_to - self.y_from + 17
            else:
                # West move
                return self.y_from - self.y_to + 25
        elif self.y_to - self.y_from == 1:
            # Short Diagonal East
            if self.x_to - self.x_from == 1:
                return 34
            elif self.x_to - self.x_from == -1:
                return 35
            # Horse Vertical East
            elif self.x_to > self.x_from:
                return 42
            else:
                return 43
        elif self.y_to - self.y_from == -1:
            # Short Diagonal West
            if self.x_to - self.x_from == 1:
                return 36
            elif self.x_to - self.x_from == -1:
                return 37
            # Horse Vertical West
            elif self.x_to > self.x_from:
                return 44
            else:
                return 45
        elif self.x_to - self.x_from == 1:
            # Horse Horizontal North
            if self.y_to > self.y_from:
                return 46
            else:
                return 47
        elif self.x_to - self.x_from == -1:
            # Horse Horizontal South
            if self.y_to > self.y_from:
                return 48
            else:
                return 49
        elif self.y_to - self.y_from == 2:
            # Large Diagonal East
            if self.x_to - self.x_from == 2:
                return 38
            elif self.x_to - self.x_from == -2:
                return 39
            # Elephant Vertical East
            elif self.x_to > self.x_from:
                return 50
            else:
                return 51
        elif self.y_to - self.y_from == -2:
            # Large Diagonal West
            if self.x_to - self.x_from == 2:
                return 40
            elif self.x_to - self.x_from == -2:
                return 41
            # Elephant Vertical West
            elif self.x_to > self.x_from:
                return 52
            else:
                return 53
        elif self.x_to - self.x_from == 2:
            # Elephant Horizontal North
            if self.y_to > self.y_from:
                return 54
            else:
                return 55
        elif self.x_to - self.x_from == -2:
            # Horse Horizontal South
            if self.y_to > self.y_from:
                return 56
            else:
                return 57
