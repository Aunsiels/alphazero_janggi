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

    def get_features(self, symmetry=False):
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
                if not symmetry:
                    return self.y_to - self.y_from + 17
                else:
                    return self.y_to - self.y_from + 25
            else:
                # West move
                if not symmetry:
                    return self.y_from - self.y_to + 25
                else:
                    return self.y_from - self.y_to + 17
        elif self.y_to - self.y_from == 1:
            # Short Diagonal East
            if self.x_to - self.x_from == 1:
                if not symmetry:
                    return 34
                else:
                    return 36
            elif self.x_to - self.x_from == -1:
                if not symmetry:
                    return 35
                else:
                    return 37
            # Horse Vertical East
            elif self.x_to > self.x_from:
                if not symmetry:
                    return 42
                else:
                    return 44
            else:
                if not symmetry:
                    return 43
                else:
                    return 45
        elif self.y_to - self.y_from == -1:
            # Short Diagonal West
            if self.x_to - self.x_from == 1:
                if not symmetry:
                    return 36
                else:
                    return 34
            elif self.x_to - self.x_from == -1:
                if not symmetry:
                    return 37
                else:
                    return 35
            # Horse Vertical West
            elif self.x_to > self.x_from:
                if not symmetry:
                    return 44
                else:
                    return 42
            else:
                if not symmetry:
                    return 45
                else:
                    return 43
        elif self.x_to - self.x_from == 1:
            # Horse Horizontal North
            if self.y_to > self.y_from:
                if not symmetry:
                    return 46
                else:
                    return 47
            else:
                if not symmetry:
                    return 47
                else:
                    return 46
        elif self.x_to - self.x_from == -1:
            # Horse Horizontal South
            if self.y_to > self.y_from:
                if not symmetry:
                    return 48
                else:
                    return 49
            else:
                if not symmetry:
                    return 49
                else:
                    return 48
        elif self.y_to - self.y_from == 2:
            # Large Diagonal East
            if self.x_to - self.x_from == 2:
                if not symmetry:
                    return 38
                else:
                    return 40
            elif self.x_to - self.x_from == -2:
                if not symmetry:
                    return 39
                else:
                    return 41
            # Elephant Vertical East
            elif self.x_to > self.x_from:
                if not symmetry:
                    return 50
                else:
                    return 52
            else:
                if not symmetry:
                    return 51
                else:
                    return 53
        elif self.y_to - self.y_from == -2:
            # Large Diagonal West
            if self.x_to - self.x_from == 2:
                if not symmetry:
                    return 40
                else:
                    return 38
            elif self.x_to - self.x_from == -2:
                if not symmetry:
                    return 41
                else:
                    return 39
            # Elephant Vertical West
            elif self.x_to > self.x_from:
                if not symmetry:
                    return 52
                else:
                    return 50
            else:
                if not symmetry:
                    return 53
                else:
                    return 51
        elif self.x_to - self.x_from == 2:
            # Elephant Horizontal North
            if self.y_to > self.y_from:
                if not symmetry:
                    return 54
                else:
                    return 55
            else:
                if not symmetry:
                    return 55
                else:
                    return 54
        elif self.x_to - self.x_from == -2:
            # Horse Horizontal South
            if self.y_to > self.y_from:
                if not symmetry:
                    return 56
                else:
                    return 57
            else:
                if not symmetry:
                    return 57
                else:
                    return 56
