import time

from janggi.board import Board
from janggi.utils import Color


class Game:

    def __init__(self, player_blue, player_red, board):
        self.player_blue = player_blue
        self.player_blue.game = self
        self.player_red = player_red
        self.player_red.game = self
        self.current_player = Color.BLUE
        self.board = board
        self.round = 0
        self.actions = []

    def run_game(self, iter_max=-1):
        while not self.is_finished(iter_max):
            # print(self.round, self.current_player)
            # begin_time = time.time()
            new_action = self.get_next_action()
            self.actions.append(new_action)
            self.board.apply_action(new_action)
            self.switch_player()
            self.board.invalidate_action_cache(new_action)  # Try to reduce memory usage
            self.round += 1
            #print(time.time() - begin_time)
            # print(self.board)
        if not self.board.is_finished(self.current_player):
            score_BLUE = self.board.get_score(Color.BLUE)
            score_RED = self.board.get_score(Color.RED)
            if score_BLUE > score_RED:
                return Color.BLUE
            else:
                return Color.RED
        return Color(-self.current_player.value)

    def switch_player(self):
        self.current_player = Color(-self.current_player.value)

    def get_next_action(self):
        if self.current_player == Color.BLUE:
            new_action = self.player_blue.play_action()
        else:
            new_action = self.player_red.play_action()
        return new_action

    def get_current_actions(self):
        return self.board.get_actions(self.current_player)

    def is_finished(self, iter_max=200):
        if self.actions:
            return self.board.is_finished(self.current_player, self.actions[-1]) or self.round == iter_max
        else:
            return self.board.is_finished(self.current_player) or self.round == iter_max

    def get_reward(self):
        score_BLUE = self.board.get_score(Color.BLUE)
        score_RED = self.board.get_score(Color.RED)
        if score_BLUE > score_RED:
            if self.current_player == Color.BLUE:
                return 1
            else:
                return -1
        else:
            if self.current_player == Color.BLUE:
                return -1
            else:
                return 1

    def get_features(self):
        return self.board.get_features(self.current_player, self.round)
