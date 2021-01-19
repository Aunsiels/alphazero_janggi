import random


class Player:

    def __init__(self, color):
        self.color = color
        self.game = None

    def play_action(self):
        raise NotImplementedError

    def think(self):
        pass

    def stop_thinking(self):
        pass


class RandomPlayer(Player):

    def play_action(self):
        actions = self.game.board.get_actions(self.color)
        if actions:
            return random.choice(actions)
        return None
