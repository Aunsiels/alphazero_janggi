import json
import time

from janggi.action import Action
from janggi.board import Board
from janggi.utils import Color


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
            action = Action.from_uci_usi(move)
            game.apply_action(action)
        return game

    def to_uci_usi(self):
        board_temp = Board(self.board.start_blue, self.board.start_red)
        res_l = ["position", "fen", board_temp.to_fen(Color.BLUE, 0),
                 "moves"]
        for action in self.actions:
            if action is None:
                res_l.append("a1a1")
            else:
                res_l.append(action.to_uci_usi())
        return " ".join(res_l)

    def run_game(self, iter_max=-1):
        begin_game_time = time.time()
        while not self.is_finished(iter_max):
            # print(self.round, self.current_player)
            # begin_time = time.time()
            new_action = self.get_next_action()
            self.apply_action(new_action)
            # print(new_action, new_action.eaten)
            # print(self.current_player, self.board.get_score(self.current_player))
            # print(time.time() - begin_time)
            # print(repr(self.board))
            # print(self.board)
            # print(new_action)
        end_game_time = time.time()
        print("Mean time per action", (end_game_time - begin_game_time) / self.round)
        is_finished = self.board.is_finished(self.current_player)
        if self.round == iter_max:
            print("Max round win")
        elif not is_finished:
            print("Score too low win.")
        if not is_finished:
            score_BLUE = self.board.get_score(Color.BLUE)
            score_RED = self.board.get_score(Color.RED)
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

    def to_json(self, mcts_node=None):
        result = dict()
        result["initial_fen"] = Board(self.board.start_blue, self.board.start_red).to_fen(Color.BLUE, 0)
        winner = self.get_winner()
        if winner == Color.BLUE:
            result["winner"] = "BLUE"
        else:
            result["winner"] = "RED"
        result["moves"] = []
        current_node = mcts_node
        for action in self.actions:
            temp = dict()
            temp["played"] = action.to_uci_usi()
            if current_node is not None:
                temp["total_N"] = current_node.total_N
                temp["N"] = dict()
                for action_mcts, value in current_node.N.items():
                    temp["N"][action_mcts.to_uci_usi()] = value
                if action in current_node.next_nodes:
                    current_node = current_node.next_nodes[action]
                else:
                    current_node = None
            else:
                temp["total_N"] = 1
                temp["N"] = {action.to_uci_usi(): 1}
            result["moves"].append(temp)
        return json.dumps(result)
