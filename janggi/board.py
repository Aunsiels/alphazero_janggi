import itertools

import torch
from janggi.piece import Soldier, Cannon, General, Chariot, Elephant, Horse, Guard
from janggi.utils import BOARD_HEIGHT, BOARD_WIDTH, Color


class Board:

    def __init__(self, start_blue="yang", start_red="yang"):
        self.board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self._blue_general = None
        self._red_general = None
        self.start_blue = start_blue
        self.start_red = start_red
        self._initialize_pieces()
        self._current_action_cache_node = ActionCacheNode(None)
        self._blue_pieces = []
        self._red_pieces = []
        self._initialise_pieces_per_color()

    def _initialise_pieces_per_color(self):
        self._blue_pieces = []
        self._red_pieces = []
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                value = self.get(x, y)
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

    def _initialize_pieces(self):
        self._initialize_soldiers()
        self._initialize_cannons()
        self._initialize_general()
        self._initialize_chariots()
        self._initialise_elephants()
        self._initialise_horses()
        self._initialize_guards()

    def _initialize_guards(self):
        self.set(0, 3, Guard(0, 3, Color.BLUE, self))
        self.set(0, 5, Guard(0, 5, Color.BLUE, self))
        self.set(9, 3, Guard(9, 3, Color.RED, self))
        self.set(9, 5, Guard(9, 5, Color.RED, self))

    def _initialise_horses(self):
        if self.start_blue == "won" or self.start_blue == "sang":
            self.set(0, 1, Horse(0, 1, Color.BLUE, self))
        else:
            self.set(0, 2, Horse(0, 2, Color.BLUE, self))
        if self.start_blue == "won" or self.start_blue == "gwee":
            self.set(0, 7, Horse(0, 7, Color.BLUE, self))
        else:
            self.set(0, 6, Horse(0, 6, Color.BLUE, self))

        if self.start_red == "won" or self.start_red == "sang":
            self.set(9, 1, Horse(9, 1, Color.RED, self))
        else:
            self.set(9, 2, Horse(9, 2, Color.RED, self))
        if self.start_red == "won" or self.start_red == "gwee":
            self.set(9, 7, Horse(9, 7, Color.RED, self))
        else:
            self.set(9, 6, Horse(9, 6, Color.RED, self))

    def _initialise_elephants(self):
        if self.start_blue == "won" or self.start_blue == "sang":
            self.set(0, 2, Elephant(0, 2, Color.BLUE, self))
        else:
            self.set(0, 1, Elephant(0, 1, Color.BLUE, self))
        if self.start_blue == "won" or self.start_blue == "gwee":
            self.set(0, 6, Elephant(0, 6, Color.BLUE, self))
        else:
            self.set(0, 7, Elephant(0, 7, Color.BLUE, self))

        if self.start_red == "won" or self.start_red == "sang":
            self.set(9, 2, Elephant(9, 2, Color.RED, self))
        else:
            self.set(9, 1, Elephant(9, 1, Color.RED, self))
        if self.start_red == "won" or self.start_red == "gwee":
            self.set(9, 6, Elephant(9, 6, Color.RED, self))
        else:
            self.set(9, 7, Elephant(9, 7, Color.RED, self))

    def _initialize_chariots(self):
        self.set(0, 0, Chariot(0, 0, Color.BLUE, self))
        self.set(0, 8, Chariot(0, 8, Color.BLUE, self))
        self.set(9, 0, Chariot(9, 0, Color.RED, self))
        self.set(9, 8, Chariot(9, 8, Color.RED, self))

    def _initialize_general(self):
        self.set(1, 4, General(1, 4, Color.BLUE, self))
        self._blue_general = self.get(1, 4)
        self.set(8, 4, General(8, 4, Color.RED, self))
        self._red_general = self.get(8, 4)

    def _initialize_cannons(self):
        self.set(2, 1, Cannon(2, 1, Color.BLUE, self))
        self.set(2, 7, Cannon(2, 7, Color.BLUE, self))
        self.set(7, 1, Cannon(7, 1, Color.RED, self))
        self.set(7, 7, Cannon(7, 7, Color.RED, self))

    def _initialize_soldiers(self):
        for y in range(0, BOARD_WIDTH, 2):
            self.set(3, y, Soldier(3, y, Color.BLUE, self))
        for y in range(0, BOARD_WIDTH, 2):
            self.set(6, y, Soldier(6, y, Color.RED, self))

    def __str__(self):
        representation = []
        for x in range(BOARD_HEIGHT - 1, -1, -1):
            to_print = []
            for y in range(BOARD_WIDTH):
                if self.get(x, y) is None:
                    to_print.append(".")
                else:
                    to_print.append(str(self.get(x, y)))
            representation.append(" ".join(to_print))
        return "\n".join(representation) + "\n"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    @staticmethod
    def is_in(x, y):
        return 0 <= x < BOARD_HEIGHT and 0 <= y < BOARD_WIDTH

    def get_actions(self, color, previous_actions=None):
        # Check if in cache
        if color == Color.RED:
            if self._current_action_cache_node.next_actions_red is not None:
                return self._current_action_cache_node.next_actions_red
        else:
            if self._current_action_cache_node.next_actions_blue is not None:
                return self._current_action_cache_node.next_actions_blue

        if color == Color.BLUE:
            pieces = self._blue_pieces
        else:
            pieces = self._red_pieces
        actions_list = []
        filtered_pieces = [piece for piece in pieces if piece.is_alive]

        for piece in filtered_pieces:
            actions_list.append(piece.get_actions())
        unfiltered_actions = list(itertools.chain(*actions_list))

        if color == Color.BLUE:
            other_pieces = self._red_pieces
            general = self._blue_general
        else:
            other_pieces = self._blue_pieces
            general = self._red_general
        potential_threads = [piece
                             for piece in other_pieces
                             if piece.is_potentially_threatening(general.x, general.y)]
        # Exclude actions creating a check
        filtered_actions = []
        for action in unfiltered_actions:
            if previous_actions is not None and is_repetition(previous_actions, action):
                continue
            self.apply_action(action)
            if action.x_to == general.x and action.y_to == general.y:
                # If we are the general, we have no choice
                if not self.is_check(color):
                    filtered_actions.append(action)
            else:
                found = False
                for potential_thread in potential_threads:
                    if not potential_thread.is_alive:
                        continue
                    other_actions = potential_thread.get_actions()
                    for other_action in other_actions:
                        if other_action.x_to == general.x and other_action.y_to == general.y:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    filtered_actions.append(action)

            self.reverse_action(action)
        actions = filtered_actions

        # Put in cache
        if color == Color.RED:
            self._current_action_cache_node.next_actions_red = actions
        else:
            self._current_action_cache_node.next_actions_blue = actions

        return actions

    def is_finished(self, color, last_action=None):
        score = self.get_score(color)
        return score == 0 or \
               (score < 20 and last_action is not None and last_action.eaten is None) or \
               (len(self.get_actions(color)) == 0 and self.is_check(color))

    def is_check(self, color):
        if color == Color.BLUE:
            king_x = self._blue_general.x
            king_y = self._blue_general.y
            other_pieces = self._red_pieces
        else:
            king_x = self._red_general.x
            king_y = self._red_general.y
            other_pieces = self._blue_pieces
        for piece in other_pieces:
            if not piece.is_alive or not piece.is_potentially_threatening(king_x, king_y):
                continue
            actions = piece.get_actions()
            for action in actions:
                if action.x_to == king_x and action.y_to == king_y:
                    return True
        return False

    def apply_action(self, action):
        if action not in self._current_action_cache_node.next_nodes:
            self._current_action_cache_node.next_nodes[action] = ActionCacheNode(self._current_action_cache_node)
        self._current_action_cache_node = self._current_action_cache_node.next_nodes[action]

        if action is None:
            # We do nothing
            return

        piece_from = self.get(action.x_from, action.y_from)
        piece_from.x = action.x_to
        piece_from.y = action.y_to
        action.eaten = self.get(action.x_to, action.y_to)
        if action.eaten is not None:
            action.eaten.is_alive = False
        self.set(action.x_to, action.y_to, piece_from)
        self.set(action.x_from, action.y_from, None)

    def reverse_action(self, action):
        self._current_action_cache_node = self._current_action_cache_node.parent

        if action is None:
            # We do nothing
            return

        self.get(action.x_to, action.y_to).x = action.x_from
        self.get(action.x_to, action.y_to).y = action.y_from
        self.set(action.x_from, action.y_from, self.get(action.x_to, action.y_to))
        self.set(action.x_to, action.y_to, action.eaten)
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
            new_x = BOARD_HEIGHT - 1 - x
            new_y = BOARD_WIDTH - 1 - y
            return self.board[new_x][new_y]
        return self.board[x][y]

    def set(self, x, y, new_value):
        self.board[x][y] = new_value

    def get_features(self, color, n_round, data_augmentation=False):
        is_reversed = color != Color.BLUE
        # 7 pieces, for two colors, + one plan color + one plan number played
        features = torch.zeros((7 * 2 + 2, BOARD_HEIGHT, BOARD_WIDTH))
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                if data_augmentation:
                    new_y = BOARD_WIDTH - 1 - y
                else:
                    new_y = y
                current = self.get(x, new_y, is_reversed)
                if current is None:
                    continue
                features[current.get_index() + 7 * (current.color != color), x, y] = 1
        if color == Color.RED:
            features[7 * 2, :, :] = 1
        features[7 * 2 + 1, :, :] = n_round
        # features = features.to(DEVICE)
        return features


def get_action_piece(piece):
    return piece.get_actions()


def is_repetition(previous_actions, current_action):
    return len(previous_actions) > 6 and \
           current_action == previous_actions[-2] and \
           current_action == previous_actions[-4] and \
           previous_actions[-1] == previous_actions[-3] and \
           previous_actions[-3] == previous_actions[-5]


class ActionCacheNode:

    def __init__(self, parent, next_actions_blue=None,
                 next_actions_red=None):
        self.parent = parent
        self.next_nodes = dict()
        self.next_actions_blue = next_actions_blue
        self.next_actions_red = next_actions_red
