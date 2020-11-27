import random

import torch

from ia.janggi_network import JanggiNetwork
from ia.mcts import MCTS, MCTSNode
from janggi.board import Board
from janggi.game import Game
from janggi.player import Player
from janggi.utils import Color, DEVICE, get_symmetries

DEFAULT_TEMPERATURE_END = 1

DEFAULT_TEMPERATURE_THRESHOLD = 30

DEFAULT_TEMPERATURE_START = 1

DEFAULT_N_SIMULATIONS = 800

DEFAULT_C_PUCT = 4


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

    def __init__(self, color, c_puct=DEFAULT_C_PUCT, n_simulations=DEFAULT_N_SIMULATIONS, current_node=None,
                 temperature_start=DEFAULT_TEMPERATURE_START, temperature_threshold=DEFAULT_TEMPERATURE_THRESHOLD,
                 temperature_end=DEFAULT_TEMPERATURE_END):
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


class NNPlayer(RandomMCTSPlayer):

    def __init__(self, color, c_puct=DEFAULT_C_PUCT, n_simulations=DEFAULT_N_SIMULATIONS, current_node=None,
                 janggi_net=None,
                 temperature_start=DEFAULT_TEMPERATURE_START, temperature_threshold=DEFAULT_TEMPERATURE_THRESHOLD,
                 temperature_end=DEFAULT_TEMPERATURE_END):
        super().__init__(color, c_puct, n_simulations, current_node,
                         temperature_start, temperature_threshold, temperature_end)
        self.janggi_net = janggi_net or JanggiNetwork()
        if isinstance(self.janggi_net, JanggiNetwork):
            self._is_predictor = True
        else:
            self._is_predictor = False

    def predict(self):
        actions = self.game.get_current_actions()
        features = self.game.get_features()
        features = torch.unsqueeze(features, 0)
        if self._is_predictor:
            features = features.to(DEVICE)
        symm_x, symm_y = get_symmetries(self.game.current_player)
        with torch.no_grad():
            policy, value = self.janggi_net(features)
            actions_proba = dict()
            total = 0
            for action in actions:
                value_policy_action = policy[0, action.get_features(symm_x, symm_y), action.get_x_from(symm_x),
                                             action.get_y_from(symm_y)].detach().item()
                total += value_policy_action
            if total != 0:
                for action in actions:
                    value_policy_action = policy[0, action.get_features(symm_x, symm_y), action.get_x_from(symm_x),
                                                 action.get_y_from(symm_y)].detach().item()
                    actions_proba[action] = value_policy_action / total
            else:
                for action in actions:
                    value_policy_action = policy[0, action.get_features(symm_x, symm_y), action.get_x_from(symm_x),
                                                 action.get_y_from(symm_y)].detach().item()
                    actions_proba[action] = value_policy_action
            value = value[0, 0].detach().item()
        return actions_proba, value


def run_a_game():
    player_blue = NNPlayer(Color.BLUE, n_simulations=100)
    play_againt_normal(player_blue, 100, 200)


def play_againt_normal(player_blue, n_simulations, iter_max):
    player_red = RandomMCTSPlayer(Color.RED, n_simulations=n_simulations)
    fight(player_blue, player_red, iter_max)


def fight(player_blue, player_red, iter_max):
    start_blue = random.choice(["won", "sang", "yang", "gwee"])
    start_red = random.choice(["won", "sang", "yang", "gwee"])
    board = Board(start_blue=start_blue, start_red=start_red)
    game = Game(player_blue, player_red, board)
    winner = game.run_game(iter_max)
    print("Winner:", winner)
    print("Score BLUE:", board.get_score(Color.BLUE))
    print("Score RED:", board.get_score(Color.RED))
    print(board)
    return winner


if __name__ == "__main__":
    keep_going = True
    while keep_going:
        run_a_game()
        keep_going = False
