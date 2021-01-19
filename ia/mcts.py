import math
import random

import torch
import numpy as np

from janggi.parameters import DIRICHLET_ALPHA, DIRICHLET_EPSILON
from janggi.utils import get_symmetries


class MCTSNode:

    def __init__(self, is_initial=False):
        self.probabilities = None
        self.current_player = None
        self.q = dict()
        self.N = dict()
        self.next_nodes = dict()
        self.is_initial = is_initial
        self.total_N = 0
        self.predicted_value = 0

    def set_up(self, probabiliies, current_player, actions, predicted_value):
        self.predicted_value = predicted_value
        self.probabilities = probabiliies
        if self.is_initial:
            dirichlet_noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(self.probabilities))
            for i, action in enumerate(self.probabilities):
                self.probabilities[action] = (1 - DIRICHLET_EPSILON) * self.probabilities[action] + \
                                             DIRICHLET_EPSILON * dirichlet_noise[i]
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

    def get_policy(self, current_player, data_augmentation=False):
        policy = torch.zeros((58, 10, 9))
        symmetry_x, symmetry_y = get_symmetries(current_player, data_augmentation)
        for action, value in self.N.items():
            if action is None or action.is_pass():
                continue
            total_temp = self.total_N
            if total_temp != 0:
                policy[action.get_features(symmetry_x, symmetry_y),
                       action.get_x_from(symmetry_x), action.get_y_from(symmetry_y)] = value / total_temp
        return policy


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

        if current_node.probabilities is None:
            possible_actions = game.get_current_actions()
            probabilities, predicted_value = predictor.predict(game, possible_actions)
            current_node.set_up(probabilities, game.current_player, possible_actions, predicted_value)
            return -predicted_value
        else:
            possible_actions = list(current_node.q.keys())

        random.shuffle(possible_actions)

        u_max, best_action = -float("inf"), None
        for action in possible_actions:
            if action is None:
                continue
            try:
                u = current_node.q[action] + \
                    self.c_puct * current_node.probabilities[action] * \
                    math.sqrt(current_node.total_N) / (1 + current_node.N[action])
            except:
                print(current_node.probabilities)
                print(repr(game.board))
                print(current_node.total_N)
                raise
            if u > u_max:
                u_max = u
                best_action = action
        # Best action is None when there is no legal move

        game.apply_action(best_action, invalidate_cache=False)
        if best_action not in current_node.next_nodes:
            next_node = MCTSNode()
            current_node.next_nodes[best_action] = next_node
        else:
            next_node = current_node.next_nodes[best_action]
        value = self.run_simulation(next_node, game, predictor)
        game.reverse_action()

        # Might be a problem if not enough simulations
        current_node.q[best_action] = (current_node.N[best_action] * current_node.q[best_action] + value) \
                                      / (current_node.N[best_action] + 1)
        current_node.N[best_action] += 1
        current_node.total_N += 1

        return -value

    def choose_action(self, current_node, game, predictor):
        for _ in range(self.n_simulations - current_node.total_N + 1):
            self.run_simulation(current_node, game, predictor)
        items = list(current_node.N.items())
        total = current_node.total_N
        if game.round > self.temperature_threshold:
            inv_temperature = 1 / self.temperature_end
        else:
            inv_temperature = 1 / self.temperature_start
        if items:
            proba = [(x[1] / total) ** inv_temperature for x in items]
            total = sum(proba)
            for i in range(len(proba)):
                proba[i] /= total
            action = random.choices([x[0] for x in items], proba)[0]
        else:
            action = None
        return action
