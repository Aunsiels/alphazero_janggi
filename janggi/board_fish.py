import pyffish

from janggi.action import Action
from janggi.board import Board


class BoardFish:
    # Written for the purpose of comparison. Right now, not fast enough.

    def __init__(self, start_blue="yang", start_red="yang"):
        self.start_blue = start_blue
        self.start_red = start_red
        if self.start_red == "yang" or self.start_red == "sang":
            left_red = "bn"
        else:
            left_red = "nb"
        if self.start_red == "won" or self.start_red == "sang":
            right_red = "bn"
        else:
            right_red = "nb"
        if self.start_blue == "yang" or self.start_blue == "gwee":
            left_blue = "BN"
        else:
            left_blue = "NB"
        if self.start_blue == "won" or self.start_blue == "gwee":
            right_blue = "BN"
        else:
            right_blue = "NB"
        self.initial_fen = \
            "r%sa1a%sr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/R%sA1A%sR w - - 0 1" % (left_red, right_red,
                                                                                           left_blue, right_blue)
        self.previous_moves = []

    def get_actions(self, color):
        moves = pyffish.legal_moves("janggi", self.initial_fen, self.previous_moves)
        return [Action.from_uci_usi(move) for move in moves]

    def is_finished(self, color, last_action=None):
        immediate_end, result = pyffish.is_immediate_game_end("janggi", self.initial_fen, self.previous_moves)
        return immediate_end

    def get_score(self, color):
        return 60

    def apply_action(self, action):
        if action is not None:
            self.previous_moves.append(action.to_uci_usi())
        else:
            self.previous_moves.append(None)

    def get_features(self, color):
        self.previous_moves.pop()

    def __repr__(self):
        return repr(Board.from_fen(str(self)))

    def __str__(self):
        return pyffish.get_fen("janggi", self.initial_fen, self.previous_moves)

    def invalidate_action_cache(self, action):
        pass