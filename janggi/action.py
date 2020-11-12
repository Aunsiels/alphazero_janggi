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
        features_temp = np.zeros(58)
        if self.y_from == self.y_to:
            # Vertical move
            if self.x_to > self.x_from:
                # North move
                features_temp[self.x_to - self.x_from - 1] = 1
            else:
                # South move
                features_temp[self.x_from - self.x_to + 8] = 1
        elif self.x_from == self.x_to:
            # Horizontal move
            if self.y_to > self.y_from:
                # East move
                features_temp[self.y_to - self.y_from + 17] = 1
            else:
                # West move
                features_temp[self.y_from - self.y_to + 25] = 1
        elif self.y_to - self.y_from == 1:
            # Short Diagonal East
            if self.x_to - self.x_from == 1:
                features_temp[34] = 1
            elif self.x_to - self.x_from == -1:
                features_temp[35] = 1
            # Horse Vertical East
            elif self.x_to > self.x_from:
                features_temp[42] = 1
            else:
                features_temp[43] = 1
        elif self.y_to - self.y_from == -1:
            # Short Diagonal West
            if self.x_to - self.x_from == 1:
                features_temp[36] = 1
            elif self.x_to - self.x_from == -1:
                features_temp[37] = 1
            # Horse Vertical West
            elif self.x_to > self.x_from:
                features_temp[44] = 1
            else:
                features_temp[45] = 1
        elif self.x_to - self.x_from == 1:
            # Horse Horizontal North
            if self.y_to > self.y_from:
                features_temp[46] = 1
            else:
                features_temp[47] = 1
        elif self.x_to - self.x_from == -1:
            # Horse Horizontal South
            if self.y_to > self.y_from:
                features_temp[48] = 1
            else:
                features_temp[49] = 1
        elif self.y_to - self.y_from == 2:
            # Large Diagonal East
            if self.x_to - self.x_from == 2:
                features_temp[38] = 1
            elif self.x_to - self.x_from == -2:
                features_temp[39] = 1
            # Elephant Vertical East
            elif self.x_to > self.x_from:
                features_temp[50] = 1
            else:
                features_temp[51] = 1
        elif self.y_to - self.y_from == -2:
            # Large Diagonal West
            if self.x_to - self.x_from == 2:
                features_temp[40] = 1
            elif self.x_to - self.x_from == -2:
                features_temp[41] = 1
            # Elephant Vertical West
            elif self.x_to > self.x_from:
                features_temp[52] = 1
            else:
                features_temp[53] = 1
        elif self.x_to - self.x_from == 2:
            # Elephant Horizontal North
            if self.y_to > self.y_from:
                features_temp[54] = 1
            else:
                features_temp[55] = 1
        elif self.x_to - self.x_from == -2:
            # Horse Horizontal South
            if self.y_to > self.y_from:
                features_temp[56] = 1
            else:
                features_temp[57] = 1
        return features_temp
