# From https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
from janggi.utils import BOARD_HEIGHT, BOARD_WIDTH, Color
from janggi.action import Action


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Piece:

    def __init__(self, x, y, color, board):
        self.x = x
        self.y = y
        self.color = color
        self.board = board
        self.is_alive = True

    def get_actions(self):
        raise NotImplementedError

    def get_colored_str(self, value):
        from janggi.board import Color
        if self.color == Color.BLUE:
            return bcolors.OKBLUE + str(value) + bcolors.ENDC
        else:
            return bcolors.FAIL + str(value) + bcolors.ENDC

    def get_points(self):
        raise NotImplementedError

    def get_index(self):
        raise NotImplementedError


class Soldier(Piece):

    def get_points(self):
        return 2

    def get_index(self):
        return 0

    def __str__(self):
        return self.get_colored_str("S")

    def get_actions(self):
        actions = []
        left = self.y - 1
        if 0 <= left:
            value_left = self.board.get(self.x, left)
            if value_left is None or value_left.color != self.color:
                actions.append(Action(self.x, self.y, self.x, left))
        right = self.y + 1
        if right < BOARD_WIDTH:
            value_right = self.board.get(self.x, right)
            if value_right is None or value_right.color != self.color:
                actions.append(Action(self.x, self.y, self.x, right))
        top = self.x + self.color.value
        if 0 <= top < BOARD_HEIGHT:
            value_top = self.board.get(top, self.y)
            if value_top is None or value_top.color != self.color:
                actions.append(Action(self.x, self.y, top, self.y))
            if self.color == Color.BLUE:
                is_mid = self.x == 8 and self.y == 4
                can_diagonal_right = (self.x == 7 and self.y == 3) or is_mid
                can_diagonal_left = (self.x == 7 and self.y == 5) or is_mid
            else:
                is_mid = self.x == 1 and self.y == 4
                can_diagonal_right = (self.x == 2 and self.y == 3) or is_mid
                can_diagonal_left = (self.x == 3 and self.y == 5) or is_mid
            if can_diagonal_right:
                value_top_right = self.board.get(top, right)
                if value_top_right is None or value_top_right.color != self.color:
                    actions.append(Action(self.x, self.y, top, right))
            if can_diagonal_left:
                value_top_left = self.board.get(top, left)
                if value_top_left is None or value_top_left.color != self.color:
                    actions.append(Action(self.x, self.y, top, left))
        return actions


class Cannon(Piece):

    def get_points(self):
        return 7

    def get_index(self):
        return 1

    def __str__(self):
        return self.get_colored_str("C")

    def get_actions(self):
        actions = []
        self._get_actions_one_direction(actions, range(self.x + 1, BOARD_HEIGHT), [self.y])
        self._get_actions_one_direction(actions, range(self.x - 1, -1, -1), [self.y])
        self._get_actions_one_direction(actions, [self.x], range(self.y + 1, BOARD_WIDTH))
        self._get_actions_one_direction(actions, [self.x], range(self.y - 1, -1, -1))
        self._get_diagonal_actions(actions, 1, 4)
        self._get_diagonal_actions(actions, 8, 4)
        return actions

    def _get_diagonal_actions(self, actions, center_x, center_y):
        for x_diff in [-1, 1]:
            for y_diff in [-1, 1]:
                is_in_diagonal_fortress = (self.x - x_diff == center_x and self.y - y_diff == center_y)
                if not is_in_diagonal_fortress:
                    continue
                center_fortress_is_occupied = self.board.get(center_x, center_y) is not None
                if not center_fortress_is_occupied:
                    continue
                diff_center_x = center_x - x_diff
                diff_center_y = center_y - y_diff
                value = self.board.get(diff_center_x, diff_center_y)
                arrival_is_free = value is None
                can_eat_arrival = value is not None and \
                                  value.color != self.color
                arrival_is_legal = (arrival_is_free or can_eat_arrival)
                if arrival_is_legal:
                    actions.append(Action(self.x, self.y, diff_center_x, diff_center_y))

    def _get_actions_one_direction(self, actions, x_tos, y_tos):
        encounter_piece_jump = False
        for y_to in y_tos:
            for x_to in x_tos:
                value = self.board.get(x_to, y_to)
                if value is None:
                    if encounter_piece_jump:
                        actions.append(Action(self.x, self.y, x_to, y_to))
                elif value.get_index() == self.get_index():
                    return
                elif not encounter_piece_jump:
                    encounter_piece_jump = True
                elif value.color != self.color:
                    actions.append(Action(self.x, self.y, x_to, y_to))
                    return
                else:
                    return


class General(Piece):

    def get_points(self):
        return 0

    def get_index(self):
        return 2

    def __str__(self):
        return self.get_colored_str("K")

    def get_actions(self):
        actions = []
        if self.color == Color.BLUE:
            self._get_action_per_fortress(actions, 0, 2, 3, 5)
        else:
            self._get_action_per_fortress(actions, 7, 9, 3, 5)
        return actions

    def _get_action_per_fortress(self, actions, x_min, x_max, y_min, y_max):
        mid_x = x_min + 1
        mid_y = y_min + 1
        for x_diff in [-1, 0, 1]:
            for y_diff in [-1, 0, 1]:
                if x_diff == 0 and y_diff == 0:
                    continue
                if (self.x == mid_x or self.y == mid_y) and not (self.x == mid_x and self.y == mid_y):
                    if x_diff != 0 and y_diff != 0:
                        continue
                new_x = self.x + x_diff
                new_y = self.y + y_diff
                is_in_fortress = x_min <= new_x <= x_max and y_min <= new_y <= y_max
                if not is_in_fortress:
                    continue
                destination_is_legal = (self.board.get(new_x, new_y) is None or
                                        self.board.get(new_x, new_y).color != self.color)
                # will_be_check = any([action.x_to == new_x and action.y_to == new_y for action in
                #                      self.board.get_actions(Color(-self.color.value), exclude_general=True)])
                if destination_is_legal:  # and not will_be_check:
                    actions.append(Action(self.x, self.y, new_x, new_y))


class Chariot(Piece):

    def get_points(self):
        return 13

    def get_index(self):
        return 3

    def __str__(self):
        return self.get_colored_str("R")

    def get_actions(self):
        actions = []
        self._get_normal_actions_x(actions, range(self.x + 1, BOARD_HEIGHT), self.y)
        self._get_normal_actions_x(actions, range(self.x - 1, -1, -1), self.y)
        self._get_normal_actions_y(actions, self.x, range(self.y + 1, BOARD_WIDTH))
        self._get_normal_actions_y(actions, self.x, range(self.y - 1, -1, -1))
        self._get_diagonal_actions(actions, 1, 4)
        self._get_diagonal_actions(actions, 8, 4)
        return actions

    def _get_normal_actions_x(self, actions, x_tos, y_to):
        for x_to in x_tos:
            value = self.board.get(x_to, y_to)
            if value is None:
                actions.append(Action(self.x, self.y, x_to, y_to))
            elif value.color != self.color:
                actions.append(Action(self.x, self.y, x_to, y_to))
                return
            else:
                return

    def _get_normal_actions_y(self, actions, x_to, y_tos):
        for y_to in y_tos:
            value = self.board.get(x_to, y_to)
            if value is None:
                actions.append(Action(self.x, self.y, x_to, y_to))
            elif value.color != self.color:
                actions.append(Action(self.x, self.y, x_to, y_to))
                return
            else:
                return

    def _get_diagonal_actions(self, actions, center_x, center_y):
        self._get_diagonal_actions_sub(actions, center_x, center_y, -1, 1)
        self._get_diagonal_actions_sub(actions, center_x, center_y, -1, -1)
        self._get_diagonal_actions_sub(actions, center_x, center_y, 1, 1)
        self._get_diagonal_actions_sub(actions, center_x, center_y, 1, -1)
        if self.x == center_x and self.y == center_y:
            self._get_diagonal_actions_in_center(actions, center_x, center_y, -1, 1)
            self._get_diagonal_actions_in_center(actions, center_x, center_y, -1, -1)
            self._get_diagonal_actions_in_center(actions, center_x, center_y, 1, 1)
            self._get_diagonal_actions_in_center(actions, center_x, center_y, 1, -1)

    def _get_diagonal_actions_in_center(self, actions, center_x, center_y, x_diff, y_diff):
        new_x = center_x + x_diff
        new_y = center_y + y_diff
        value = self.board.get(new_x, new_y)
        if value is None or value.color != self.color:
            actions.append(Action(self.x, self.y, new_x, new_y))

    def _get_diagonal_actions_sub(self, actions, center_x, center_y, x_diff, y_diff):
        is_in_diagonal_fortress = (self.x - x_diff == center_x and self.y - y_diff == center_y)
        if not is_in_diagonal_fortress:
            return
        center_fortress_is_occupied = self.board.get(center_x, center_y) is not None
        if center_fortress_is_occupied and self.board.get(center_x, center_y) != self.color:
            actions.append(Action(self.x, self.y, center_x, center_y))
        elif not center_fortress_is_occupied:
            actions.append(Action(self.x, self.y, center_x, center_y))
            diff_center_x = center_x - x_diff
            diff_center_y = center_y - y_diff
            arrival_is_free = self.board.get(diff_center_x, diff_center_y) is None
            can_eat_arrival = self.board.get(diff_center_x, diff_center_y) is not None and \
                              self.board.get(diff_center_x, diff_center_y).color != self.color
            arrival_is_legal = (arrival_is_free or can_eat_arrival)
            if arrival_is_legal:
                actions.append(Action(self.x, self.y, diff_center_x, diff_center_y))


class Elephant(Piece):

    def get_points(self):
        return 3

    def get_index(self):
        return 4

    def __str__(self):
        return self.get_colored_str("E")

    def get_actions(self):
        actions = []
        self._get_actions_sub(actions, -1, 1)
        self._get_actions_sub(actions, -1, -1)
        self._get_actions_sub(actions, 1, 1)
        self._get_actions_sub(actions, 1, -1)
        return actions

    def _get_actions_sub(self, actions, fix_x, fix_y):
        short_x = self.x + fix_x
        long_x = self.x + 2 * fix_x
        very_long_x = self.x + 3 * fix_x
        short_y = self.y + fix_y
        long_y = self.y + 2 * fix_y
        very_long_y = self.y + 3 * fix_y

        third_jump_ok = self.board.is_in(very_long_x, long_y) and (
                self.board.get(very_long_x, long_y) is None or
                self.board.get(very_long_x, long_y).color != self.color)
        second_jump_ok = third_jump_ok and self.board.get(long_x, short_y) is None
        first_jump_ok = second_jump_ok and self.board.get(short_x, self.y) is None
        if first_jump_ok:
            actions.append(Action(self.x, self.y, very_long_x, long_y))

        third_jump_ok = self.board.is_in(long_x, very_long_y) and (
                self.board.get(long_x, very_long_y) is None or
                self.board.get(long_x, very_long_y).color != self.color)
        second_jump_ok = third_jump_ok and self.board.get(short_x, long_y) is None
        first_jump_ok = second_jump_ok and self.board.get(self.x, short_y) is None
        if first_jump_ok:
            actions.append(Action(self.x, self.y, long_x, very_long_y))


class Horse(Piece):

    def get_points(self):
        return 5

    def get_index(self):
        return 5

    def __str__(self):
        return self.get_colored_str("H")

    def get_actions(self):
        actions = []
        self._get_actions_sub(actions, -1, 1)
        self._get_actions_sub(actions, -1, -1)
        self._get_actions_sub(actions, 1, 1)
        self._get_actions_sub(actions, 1, -1)
        return actions

    def _get_actions_sub(self, actions, fix_x, fix_y):
        short_x = self.x + fix_x
        long_x = self.x + 2 * fix_x
        short_y = self.y + fix_y
        long_y = self.y + 2 * fix_y

        if self.board.is_in(long_x, short_y):
            value_long_short = self.board.get(long_x, short_y)
            second_jump_ok = value_long_short is None or value_long_short.color != self.color
            first_jump_ok = second_jump_ok and self.board.get(short_x, self.y) is None
            if first_jump_ok:
                actions.append(Action(self.x, self.y, long_x, short_y))
        if self.board.is_in(short_x, long_y):
            value_short_long = self.board.get(short_x, long_y)
            second_jump_ok = value_short_long is None or value_short_long.color != self.color
            first_jump_ok = second_jump_ok and self.board.get(self.x, short_y) is None
            if first_jump_ok:
                actions.append(Action(self.x, self.y, short_x, long_y))


class Guard(Piece):

    def get_points(self):
        return 3

    def get_index(self):
        return 6

    def __str__(self):
        return self.get_colored_str("G")

    def get_actions(self):
        actions = []
        if self.color == Color.BLUE:
            self._get_action_per_fortress(actions, 0, 2, 3, 5)
        else:
            self._get_action_per_fortress(actions, 7, 9, 3, 5)
        return actions

    def _get_action_per_fortress(self, actions, x_min, x_max, y_min, y_max):
        mid_x = x_min + 1
        mid_y = y_min + 1
        for x_diff in [-1, 0, 1]:
            for y_diff in [-1, 0, 1]:
                if x_diff == 0 and y_diff == 0:
                    continue
                if (self.x == mid_x or self.y == mid_y) and not (self.x == mid_x and self.y == mid_y):
                    if x_diff != 0 and y_diff != 0:
                        continue
                new_x = self.x + x_diff
                new_y = self.y + y_diff
                is_in_fortress = x_min <= new_x <= x_max and y_min <= new_y <= y_max
                if not is_in_fortress:
                    continue
                destination_is_legal = (self.board.get(new_x, new_y) is None or
                                        self.board.get(new_x, new_y).color != self.color)
                if destination_is_legal:  # and not will_be_check:
                    actions.append(Action(self.x, self.y, new_x, new_y))