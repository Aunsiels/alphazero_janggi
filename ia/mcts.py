import math
import random


class MCTSNode:

    def __init__(self):
        self.probabilities = None
        self.current_player = None
        self.q = dict()
        self.N = dict()
        self.next_nodes = dict()

    def set_up(self, probabiliies, current_player, actions):
        self.probabilities = probabiliies
        self.current_player = current_player
        self.q = dict()
        self.N = dict()
        self.next_nodes = dict()
        for action in actions:
            self.q[action] = 0
            self.N[action] = 0
        if not actions:
            self.q[None] = 0
            self.N[None] = 0


class MCTS:

    def __init__(self, c_puct, n_simulations):
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def run_simulation(self, current_node, game, predictor):
        if game.is_finished():
            reward = game.get_reward()
            return -reward

        if current_node.probabilities is None:
            probabilities, predicted_value = predictor.predict()
            current_node.set_up(probabilities, game.current_player, game.get_current_actions())
            return -predicted_value

        u_max, best_action = -float("inf"), None
        for action in game.get_current_actions():
            u = current_node.q[action] + \
                self.c_puct * current_node.probabilities[action] * \
                math.sqrt(sum(current_node.N.values())) / (1 + current_node.N[action])
            if u > u_max:
                u_max = u
                best_action = action
        # Best action is None when there is no legal move

        game.board.apply_action(best_action)
        game.switch_player()
        if best_action not in current_node.next_nodes:
            next_node = MCTSNode()
            current_node.next_nodes[best_action] = next_node
        else:
            next_node = current_node.next_nodes[best_action]
        value = self.run_simulation(next_node, game, predictor)
        game.board.reverse_action(best_action)
        game.switch_player()

        current_node.q[best_action] = (current_node.N[best_action] * current_node.q[best_action] + value) \
                                      / (current_node.N[best_action] + 1)
        current_node.N[best_action] += 1

        return -value

    def choose_action(self, current_node, game, predictor):
        for _ in range(self.n_simulations):
            self.run_simulation(current_node, game, predictor)
        items = list(current_node.q.items())
        random.shuffle(items)
        action = max(items, key=lambda x: x[1])[0]
        return action
