import time

from janggi.action import Action, UCI_USI_CONVERSIONS
from janggi.player import Player


class StockfishPlayer(Player):

    def __init__(self, color, process_engine, think_time=1):
        super().__init__(color)
        self.process_engine = process_engine
        self.think_time = think_time

    def play_action(self):
        self.process_engine.stdin.write("go\n")
        self.process_engine.stdin.flush()
        if self.think_time != -1:
            time.sleep(self.think_time)
            self.process_engine.stdin.write("d\n")
        self.process_engine.stdin.flush()
        while True:
            line = self.process_engine.stdout.readline()
            line = line.strip()
            if "move" in line:
                if "pass" in line:
                    return Action(0, 0, 0, 0)
                move = line.split()[1].strip()
                action = Action(int(move[1]), int(UCI_USI_CONVERSIONS[move[0]]),
                                int(move[3]), int(UCI_USI_CONVERSIONS[move[2]]))
                return action
