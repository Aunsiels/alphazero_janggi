import torch

from ia.janggi_network import JanggiNetwork
from ia.mcts import MCTS, MCTSNode
from janggi.board import Board
from janggi.game import Game
from janggi.player import Player, RandomPlayer
from janggi.utils import Color


class RandomMCTSPlayer(Player):

    def play_action(self):
        self._apply_latest_actions()
        return self.mcts.choose_action(self.current_node, self.game, self)

    def _apply_latest_actions(self):
        for i in range(self.last_action_index, len(self.game.actions)):
            action = self.game.actions[i]
            previous_node = self.current_node
            if action in self.current_node.next_nodes:
                self.current_node = self.current_node.next_nodes[action]
            else:
                self.current_node = MCTSNode()
            del previous_node
        self.last_action_index = len(self.game.actions) - 1

    def __init__(self, color, c_puct=4, n_simulations=800, current_node=None, temperature_start=1, temperature_threshold=30, temperature_end=1):
        super().__init__(color)
        self.mcts = MCTS(c_puct, n_simulations, temperature_start, temperature_threshold, temperature_end)
        self.current_node = current_node or MCTSNode()
        self.last_action_index = 0

    def predict(self):
        actions = self.game.get_current_actions()
        diff_score = (self.game.board.get_score(Color.BLUE) - self.game.board.get_score(Color.RED)) / 73.5 / 10
        if self.game.current_player == Color.RED:
            diff_score *= -1
        return {action: 1 / len(actions) for action in actions}, diff_score


if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)


class NNPlayer(RandomMCTSPlayer):

    def __init__(self, color, c_puct=4, n_simulations=800, current_node=None, janggi_net=JanggiNetwork(), temperature_start=1, temperature_threshold=30, temperature_end=1):
        super().__init__(color, c_puct, n_simulations, current_node, temperature_start, temperature_threshold, temperature_end)
        self.janggi_net = janggi_net.to(device)

    def predict(self):
        actions = self.game.get_current_actions()
        features = self.game.get_features()
        with torch.no_grad():
            policy, value = self.janggi_net(features)
            actions_proba = dict()
            for action in actions:
                actions_proba[action] = policy[0, action.get_features(), action.x_from, action.y_from].detach().item()
            value = value[0, 0].detach().item()
        return actions_proba, value


def run_a_game():
    player_blue = NNPlayer(Color.BLUE, n_simulations=100)
    play_againt_normal(player_blue, 100, 200)


def play_againt_normal(player_blue, n_simulations, iter_max):
    player_red = RandomMCTSPlayer(Color.RED, n_simulations=n_simulations)
    fight(player_blue, player_red, iter_max)


def fight(player_blue, player_red, iter_max):
    board = Board()
    game = Game(player_blue, player_red, board)
    winner = game.run_game(iter_max)
    print("Winner:", winner)
    print("Score BLUE:", board.get_score(Color.BLUE))
    print("Score RED:", board.get_score(Color.RED))
    print(board)


if __name__ == "__main__":
    keep_going = True
    while keep_going:
        run_a_game()
        keep_going = False
