import itertools

import torch
from janggi.piece import Soldier, Cannon, General, Chariot, Elephant, Horse, Guard
from janggi.utils import BOARD_HEIGHT, BOARD_WIDTH, Color


class Board:

    def __init__(self):
        self.board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self._blue_general = None
        self._red_general = None
        self._initialize_pieces()
        self._hash_cache = None
        self._str_cache = None
        self._current_action_cache_node = ActionCacheNode(None)
        self._blue_pieces = []
        self._red_pieces = []
        self._initialise_pieces_per_color()

    def _initialise_pieces_per_color(self):
        self._blue_pieces = []
        self._red_pieces = []
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                value = self.board[x][y]
                if value is not None:
                    if value.color == Color.BLUE:
                        self._blue_pieces.append(value)
                        if isinstance(value, General):
                            self._blue_general = value
                    else:
                        self._red_pieces.append(value)
                        if isinstance(value, General):
                            self._red_general = value

    def invalidate_action_cache(self, action=None):
        if action is None:
            self._current_action_cache_node = ActionCacheNode(None)
        else:
            self._current_action_cache_node.parent = None

    def _invalidate_cache(self):
        self._hash_cache = None
        self._str_cache = None

    def _initialize_pieces(self):
        self._initialize_soldiers()
        self._initialize_cannons()
        self._initialize_general()
        self._initialize_chariots()
        self._initialise_elephants()
        self._initialise_horses()
        self._initialize_guards()

    def _initialize_guards(self):
        self.board[0][3] = Guard(0, 3, Color.BLUE, self)
        self.board[0][5] = Guard(0, 5, Color.BLUE, self)
        self.board[9][3] = Guard(9, 3, Color.RED, self)
        self.board[9][5] = Guard(9, 5, Color.RED, self)

    def _initialise_horses(self):
        self.board[0][2] = Horse(0, 2, Color.BLUE, self)
        self.board[0][6] = Horse(0, 6, Color.BLUE, self)
        self.board[9][2] = Horse(9, 2, Color.RED, self)
        self.board[9][6] = Horse(9, 6, Color.RED, self)

    def _initialise_elephants(self):
        self.board[0][1] = Elephant(0, 1, Color.BLUE, self)
        self.board[0][7] = Elephant(0, 7, Color.BLUE, self)
        self.board[9][1] = Elephant(9, 1, Color.RED, self)
        self.board[9][7] = Elephant(9, 7, Color.RED, self)

    def _initialize_chariots(self):
        self.board[0][0] = Chariot(0, 0, Color.BLUE, self)
        self.board[0][8] = Chariot(0, 8, Color.BLUE, self)
        self.board[9][0] = Chariot(9, 0, Color.RED, self)
        self.board[9][8] = Chariot(9, 8, Color.RED, self)

    def _initialize_general(self):
        self.board[1][4] = General(1, 4, Color.BLUE, self)
        self._blue_general = self.board[1][4]
        self.board[8][4] = General(8, 4, Color.RED, self)
        self._red_general = self.board[8][4]

    def _initialize_cannons(self):
        self.board[2][1] = Cannon(2, 1, Color.BLUE, self)
        self.board[2][7] = Cannon(2, 7, Color.BLUE, self)
        self.board[7][1] = Cannon(7, 1, Color.RED, self)
        self.board[7][7] = Cannon(7, 7, Color.RED, self)

    def _initialize_soldiers(self):
        for y in range(0, BOARD_WIDTH, 2):
            self.board[3][y] = Soldier(3, y, Color.BLUE, self)
        for y in range(0, BOARD_WIDTH, 2):
            self.board[6][y] = Soldier(6, y, Color.RED, self)

    def __str__(self):
        if self._str_cache is None:
            representation = []
            for x in range(BOARD_HEIGHT - 1, -1, -1):
                to_print = []
                for y in range(BOARD_WIDTH):
                    if self.board[x][y] is None:
                        to_print.append(".")
                    else:
                        to_print.append(str(self.board[x][y]))
                representation.append(" ".join(to_print))
            self._str_cache = "\n".join(representation) + "\n"
        return self._str_cache

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash(str(self))
        return self._hash_cache

    def __eq__(self, other):
        return str(self) == str(other)

    @staticmethod
    def is_in(x, y):
        return 0 <= x < BOARD_HEIGHT and 0 <= y < BOARD_WIDTH

    def get_actions(self, color, exclude_general=False):
        if color == Color.RED:
            if exclude_general and self._current_action_cache_node.next_actions_no_general_red is not None:
                return self._current_action_cache_node.next_actions_no_general_red
            elif not exclude_general and self._current_action_cache_node.next_actions_red is not None:
                return self._current_action_cache_node.next_actions_red
        else:
            if exclude_general and self._current_action_cache_node.next_actions_no_general_blue is not None:
                return self._current_action_cache_node.next_actions_no_general_blue
            elif not exclude_general and self._current_action_cache_node.next_actions_blue is not None:
                return self._current_action_cache_node.next_actions_blue

        if color == Color.BLUE:
            pieces = self._blue_pieces
        else:
            pieces = self._red_pieces
        actions_list = []
        for piece in pieces:
            if piece.is_alive and (not exclude_general or not isinstance(piece, General)):
                actions_list.append(piece.get_actions())
        actions = itertools.chain(*actions_list)
        if not exclude_general:
            # Exclude actions creating a check
            filtered_actions = []
            for action in actions:
                self.apply_action(action)
                if not self.is_check(color):
                    filtered_actions.append(action)
                self.reverse_action(action)
            actions = filtered_actions

        if color == Color.RED:
            if exclude_general:
                self._current_action_cache_node.next_actions_no_general_red = actions
            else:
                self._current_action_cache_node.next_actions_red = actions
        else:
            if exclude_general:
                self._current_action_cache_node.next_actions_no_general_blue = actions
            else:
                self._current_action_cache_node.next_actions_blue = actions

        return actions

    def is_finished(self, color, last_action=None):
        return self.get_score(color) == 0 or \
               (self.get_score(color) < 20 and last_action is not None and last_action.eaten is None) or \
               (len(self.get_actions(color)) == 0 and self.is_check(color))

    def is_check(self, color, x_from=None, y_from=None, x_to=None, y_to=None):
        if color == Color.BLUE:
            king_x = self._blue_general.x
            king_y = self._blue_general.y
        else:
            king_x = self._red_general.x
            king_y = self._red_general.y
        other_actions = self.get_actions(Color(-color.value), exclude_general=True)
        return any((action.x_to == king_x and action.y_to == king_y for action in other_actions))

    def apply_action(self, action):
        self._invalidate_cache()

        if action not in self._current_action_cache_node.next_nodes:
            self._current_action_cache_node.next_nodes[action] = ActionCacheNode(self._current_action_cache_node)
        self._current_action_cache_node = self._current_action_cache_node.next_nodes[action]

        if action is None:
            # We do nothing
            return

        piece_from = self.board[action.x_from][action.y_from]
        piece_from.x = action.x_to
        piece_from.y = action.y_to
        action.eaten = self.board[action.x_to][action.y_to]
        if action.eaten is not None:
            action.eaten.is_alive = False
        self.board[action.x_to][action.y_to] = piece_from
        self.board[action.x_from][action.y_from] = None

    def reverse_action(self, action):
        self._invalidate_cache()
        self._current_action_cache_node = self._current_action_cache_node.parent

        if action is None:
            # We do nothing
            return

        self.board[action.x_to][action.y_to].x = action.x_from
        self.board[action.x_to][action.y_to].y = action.y_from
        self.board[action.x_from][action.y_from] = self.board[action.x_to][action.y_to]
        self.board[action.x_to][action.y_to] = action.eaten
        if action.eaten is not None:
            action.eaten.is_alive = True
        action.eaten = None

    def get_score(self, color):
        if color == Color.BLUE:
            score = 0
            pieces = self._blue_pieces
        else:
            score = 1.5
            pieces = self._red_pieces
        for piece in pieces:
            if piece.is_alive:
                score += piece.get_points()
        return score

    def get(self, x, y, reverse=False):
        if reverse:
            return self.board[BOARD_HEIGHT - 1 - x][BOARD_WIDTH - 1 - y]
        return self.board[x][y]

    def get_features(self, color, round):
        reversed = color != Color.BLUE
        # 7 pieces, for two colors, + one plan color + one plan number played
        features = torch.zeros((1, 7 * 2 + 2, BOARD_HEIGHT, BOARD_WIDTH))
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                current = self.get(x, y, reversed)
                if current is None:
                    continue
                features[0, current.get_index() + 7 * (current.color != color), x, y] = 1
        if color == Color.RED:
            features[0, 7 * 2, :, :] = 1
        features[0, 7 * 2 + 1, :, :] = round
        return features


class ActionCacheNode:

    def __init__(self, parent, next_actions_blue=None, next_actions_no_general_blue=None,
                 next_actions_red=None, next_actions_no_general_red=None):
        self.parent = parent
        self.next_nodes = dict()
        self.next_actions_blue = next_actions_blue
        self.next_actions_no_general_blue = next_actions_no_general_blue
        self.next_actions_red = next_actions_red
        self.next_actions_no_general_red = next_actions_no_general_red


def get_actions_piece(x):
    return x.get_actions()
