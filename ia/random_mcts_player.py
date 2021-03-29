import threading

import torch

from ia.janggi_network import JanggiNetwork
from ia.mcts import MCTS, MCTSNode
from janggi.game import Game
from janggi.parameters import DEFAULT_TEMPERATURE_END, DEFAULT_TEMPERATURE_THRESHOLD, DEFAULT_TEMPERATURE_START, \
    DEFAULT_N_SIMULATIONS, DEFAULT_C_PUCT, N_THREADS_MCTS
from janggi.player import Player
from janggi.utils import Color, DEVICE, get_symmetries, get_random_board


class RandomMCTSPlayer(Player):

    def play_action(self):
        self._apply_latest_actions()
        action = self.mcts.choose_action(self.current_node, self.game, self)
        if self.print_info:
            if self.color == Color.BLUE:
                print("Current player: Blue")
            else:
                print("Current player: RED")
            print("Current Predicted Value:", self.current_node.predicted_value)
            print("Action Played:", action.to_uci_usi())
            print("Number Simulations:", self.current_node.total_N)
            print("Top Probabilities:")
            best_actions = sorted(self.current_node.N.items(), key=lambda x: -x[1])[:10]
            for best_action, score in best_actions:
                if self.current_node.total_N != 0:
                    print(best_action.to_uci_usi(), score / self.current_node.total_N, self.current_node.q[best_action], sep="\t")
        return action

    def _apply_latest_actions(self):
        for i in range(self.last_action_index, len(self.game.actions)):
            action = self.game.actions[i]
            if action in self.current_node.next_nodes:
                self.current_node = self.current_node.next_nodes[action]
            else:
                node = MCTSNode()
                self.current_node.next_nodes[action] = node
                self.current_node = node
        self.last_action_index = len(self.game.actions)

    def __init__(self, color, c_puct=DEFAULT_C_PUCT, n_simulations=DEFAULT_N_SIMULATIONS, current_node=None,
                 temperature_start=DEFAULT_TEMPERATURE_START, temperature_threshold=DEFAULT_TEMPERATURE_THRESHOLD,
                 temperature_end=DEFAULT_TEMPERATURE_END, think_when_other=False, print_info=False):
        super().__init__(color)
        self.mcts = MCTS(c_puct, n_simulations, temperature_start, temperature_threshold, temperature_end)
        self.current_node = current_node or MCTSNode()
        self.last_action_index = 0
        self.think_when_other = think_when_other
        self.thinking_threads = None
        self.thinking_event = None
        self.print_info = print_info

    def predict(self, game, actions):
        diff_score = (game.board.get_score(Color.BLUE) - game.board.get_score(Color.RED)) / 73.5 / 2
        if game.current_player == Color.RED:
            diff_score *= -1
        return {action: 1 / len(actions) for action in actions}, diff_score

    def think(self):
        if self.think_when_other:
            self._apply_latest_actions()
            self.thinking_event = threading.Event()
            # Copy game
            games = [self.game.fake_copy() for _ in range(N_THREADS_MCTS)]
            self.thinking_threads = [ThreadKeepThinking(self.mcts, self.current_node, game, self, self.thinking_event)
                                     for game in games]
            for thread in self.thinking_threads:
                thread.start()

    def stop_thinking(self):
        if self.think_when_other and self.thinking_threads is not None:
            self.thinking_event.set()
            for thread in self.thinking_threads:
                thread.join()
            self.thinking_threads = None


class NNPlayer(RandomMCTSPlayer):

    def __init__(self, color, c_puct=DEFAULT_C_PUCT, n_simulations=DEFAULT_N_SIMULATIONS, current_node=None,
                 janggi_net=None,
                 temperature_start=DEFAULT_TEMPERATURE_START, temperature_threshold=DEFAULT_TEMPERATURE_THRESHOLD,
                 temperature_end=DEFAULT_TEMPERATURE_END, think_when_other=False, print_info=False):
        super().__init__(color, c_puct, n_simulations, current_node,
                         temperature_start, temperature_threshold, temperature_end, think_when_other, print_info)
        self.janggi_net = janggi_net or JanggiNetwork()
        if isinstance(self.janggi_net, JanggiNetwork):
            self._is_predictor = True
        else:
            self._is_predictor = False

    def predict(self, game, actions):
        features = game.get_features()
        features = torch.unsqueeze(features, 0)
        if self._is_predictor:
            features = features.to(DEVICE)
        symm_x, symm_y = get_symmetries(game.current_player)
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


class ThreadKeepThinking(threading.Thread):

    def __init__(self, mcts, current_node, game, predictor, event):
        super().__init__()
        self.mcts = mcts
        self.current_node = current_node
        self.game = game
        self.predictor = predictor
        self.event = event

    def run(self) -> None:
        while not self.event.is_set():
            self.mcts.run_simulation(self.current_node, self.game, self.predictor)


def run_a_game():
    player_blue = NNPlayer(Color.BLUE, n_simulations=100)
    play_againt_normal(player_blue, 100, 200)


def play_againt_normal(player_blue, n_simulations, iter_max):
    player_red = RandomMCTSPlayer(Color.RED, n_simulations=n_simulations)
    fight(player_blue, player_red, iter_max)


def fight(player_blue, player_red, iter_max, print_board=False):
    board = get_random_board()
    game = Game(player_blue, player_red, board)
    winner = game.run_game(iter_max, print_board=print_board)
    print("Winner:", winner)
    print("Score BLUE:", board.get_score(Color.BLUE))
    print("Score RED:", board.get_score(Color.RED))
    print(repr(board))
    print(board)
    print(game.to_uci_usi())
    return winner


if __name__ == "__main__":
    keep_going = True
    while keep_going:
        run_a_game()
        keep_going = False
