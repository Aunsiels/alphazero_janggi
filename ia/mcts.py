import math
import random

import torch
import numpy as np


if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)


class MCTSNode:

    def __init__(self, is_initial=False):
        self.probabilities = None
        self.current_player = None
        self.q = dict()
        self.N = dict()
        self.next_nodes = dict()
        self.is_initial = is_initial
        self.total_N = 0

    def set_up(self, probabiliies, current_player, actions):
        self.probabilities = probabiliies
        if self.is_initial:
            dir = np.random.dirichlet([0.3] *  len(self.probabilities))
            for i, action in enumerate(self.probabilities):
                self.probabilities[action] += dir[i]
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

    def get_policy(self):
        policy = torch.zeros((58, 10, 9))
        totals = dict()
        for action, value in self.N.items():
            if action is None:
                if None not in totals:
                    totals[None] = 0
                totals[None] += value
            else:
                if (action.x_from, action.y_from) not in totals:
                    totals[(action.x_from, action.y_from)] = 0
                totals[(action.x_from, action.y_from)] += value
        for action, value in self.N.items():
            if action is None:
                continue
            total_temp = totals[(action.x_from, action.y_from)]
            if total_temp != 0:
                policy[action.get_features(), action.x_from, action.y_from] = value / total_temp
        return policy.to(device)


class MCTS:

    def __init__(self, c_puct, n_simulations, temperature_start=1, temperature_threshold=30, temperature_end=1):
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.temperature_start = temperature_start
        self.temperature_threshold = temperature_threshold
        self.temperature_end = temperature_end

    def run_simulation(self, current_node, game, predictor):
        if game.is_finished():
            reward = game.get_reward()
            return -reward

        possible_actions = game.get_current_actions()
        if current_node.probabilities is None:
            probabilities, predicted_value = predictor.predict()
            current_node.set_up(probabilities, game.current_player, possible_actions)
            return -predicted_value

        u_max, best_action = -float("inf"), None
        for action in possible_actions:
            u = current_node.q[action] + \
                self.c_puct * current_node.probabilities[action] * \
                math.sqrt(current_node.total_N) / (1 + current_node.N[action])
            if u > u_max:
                u_max = u
                best_action = action
            # elif np.isnan(u) and best_action is None:
            #     best_action = action
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

        # Might be a problem if not enough simulations
        current_node.q[best_action] = (current_node.N[best_action] * current_node.q[best_action] + value) \
                                      / (current_node.N[best_action] + 1)
        current_node.N[best_action] += 1
        current_node.total_N += 1

        return -value

    def choose_action(self, current_node, game, predictor):
        for _ in range(self.n_simulations - current_node.total_N):
            self.run_simulation(current_node, game, predictor)
        items = list(current_node.N.items())
        if game.round > self.temperature_threshold:
            inv_temperature = 1 / self.temperature_end
        else:
            inv_temperature = 1 / self.temperature_start
        if items:
            action = random.choices([x[0] for x in items], [x[1] ** inv_temperature for x in items])[0]
        else:
            action = None
        #random.shuffle(items)
        #action = max(items, key=lambda x: x[1])[0]
        return action
