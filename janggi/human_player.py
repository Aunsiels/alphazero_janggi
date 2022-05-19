from ia.random_mcts_player import fight, RandomMCTSPlayer
from janggi.action import Action
from janggi.player import Player
from janggi.utils import Color


class HumanPlayer(Player):

    def play_action(self):
        legal_actions = self.game.get_current_actions()
        while True:
            read_data = input("Enter your action:")
            try:
                action = Action.from_uci_usi(read_data.strip())
                if action in legal_actions:
                    break
                else:
                    print("Illegal action")
            except IndexError:
                print("Invalid Action")
        return action


if __name__ == "__main__":
    player_blue = HumanPlayer(Color.BLUE)
    player_red = RandomMCTSPlayer(Color.RED, n_simulations=16000, temperature_start=0.01,
                                  temperature_threshold=30, temperature_end=0.01,
                                  think_when_other=True, print_info=True)
    winner = fight(player_blue, player_red, 200, print_board=True)
