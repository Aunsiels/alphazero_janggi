import time

from janggi.action import Action
from janggi.board import Board
from janggi.utils import Color


UCI_USI_CONVERSIONS = {
    "a": "0",
    "b": "1",
    "c": "2",
    "d": "3",
    "e": "4",
    "f": "5",
    "g": "6",
    "h": "7",
    "i": "8",
    "1": "0",
    "2": "1",
    "3": "2",
    "4": "3",
    "5": "4",
    "6": "5",
    "7": "6",
    "8": "7",
    "9": "8",
    "10": "9",
    "X": "9"
}


class Game:

    def __init__(self, player_blue, player_red, board):
        self.player_blue = player_blue
        self.player_blue.game = self
        self.player_red = player_red
        self.player_red.game = self
        self.current_player = Color.BLUE
        self.starting_fen = None
        self.board = board
        self.round = 0
        self.actions = []

    @classmethod
    def from_fen(cls, player_blue, player_red, fen):
        fen = fen.replace("--", "- -")
        board = Board.from_fen(fen)
        game = Game(player_blue, player_red, board)
        fen_split = fen.split(" ")
        if fen_split[1] == "w":
            game.current_player = Color.BLUE
        else:
            game.current_player = Color.RED
        game.round = int(fen_split[-1]) * 2 - 1
        if game.current_player == Color.BLUE:
            game.round -= 1
        game.starting_fen = fen
        # fen_split[-2] contains number of half moves
        return game

    @classmethod
    def from_uci_usi(cls, player_blue, player_red, uci_usi):
        uci_usi = uci_usi.replace("--", "- -")
        uci_usi_split = uci_usi.split(" ")
        fen = " ".join(uci_usi_split[2:8])
        moves = uci_usi_split[9:]
        game = Game.from_fen(player_blue, player_red, fen)
        for move in moves:
            move = move.replace("10", "X")
            for i in range(1, 10):
                move = move.replace(str(i), UCI_USI_CONVERSIONS[str(i)])
            for letter in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "X"]:
                move = move.replace(letter, UCI_USI_CONVERSIONS[letter])
            x_from = move[1]
            y_from = move[0]
            x_to = move[3]
            y_to = move[2]
            action = Action(int(x_from), int(y_from), int(x_to), int(y_to))
            game.apply_action(action)
        return game

    def run_game(self, iter_max=-1):
        begin_game_time = time.time()
        while not self.is_finished(iter_max):
            # print(self.round, self.current_player)
            # begin_time = time.time()
            new_action = self.get_next_action()
            # print(new_action)
            self.apply_action(new_action)
            # print(time.time() - begin_time)
            #print(repr(self.board))
            #print(self.board)
            #print(new_action)
        end_game_time = time.time()
        print("Mean time per action", (end_game_time - begin_game_time) / self.round)
        if not self.board.is_finished(self.current_player):
            score_BLUE = self.board.get_score(Color.BLUE)
            score_RED = self.board.get_score(Color.RED)
            print("Max round win")
            if score_BLUE > score_RED:
                return Color.BLUE
            else:
                return Color.RED
        else:
            print("Check win with", self.round, "rounds")
        return Color(-self.current_player.value)

    def apply_action(self, action):
        self.actions.append(action)
        self.board.apply_action(action)
        self.switch_player()
        self.board.invalidate_action_cache(action)  # Try to reduce memory usage
        self.round += 1

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

    def get_winner(self):
        if not self.board.is_finished(self.current_player):
            score_BLUE = self.board.get_score(Color.BLUE)
            score_RED = self.board.get_score(Color.RED)
            if score_BLUE > score_RED:
                winner = Color.BLUE
            else:
                winner = Color.RED
        else:
            winner = Color(-self.current_player.value)
        return winner

    def dumps(self):
        if self.starting_fen is None:
            res = [self.board.start_blue, self.board.start_red]
        else:
            res = [self.starting_fen]
        for action in self.actions:
            if action is None:
                res.append("XXXX")
            else:
                res.append(str(action.x_from) + str(action.y_from) + str(action.x_to) + str(action.y_to))
        if self.get_winner() == self.current_player:
            res.append("XXXX")
        return "\n".join(res) + "\n"
