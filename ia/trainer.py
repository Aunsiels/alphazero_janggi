import random
import time

import torch

from ia.janggi_network import JanggiLoss
from ia.mcts import MCTSNode
from ia.random_mcts_player import NNPlayer, play_againt_normal, fight, RandomMCTSPlayer
from janggi.board import Board
from janggi.game import Game
from janggi.utils import Color


class Trainer:

    def __init__(self, predictor, n_simulations=800, iter_max=200, n_simulation_opponent=800):
        self.predictor = predictor
        self.n_simulations = n_simulations
        self.iter_max = iter_max
        self.n_simulations_opponent = n_simulation_opponent

    def run_episode(self):
        examples = []
        board = Board()
        initial_node = MCTSNode(is_initial=True)
        player_blue = NNPlayer(Color.BLUE, n_simulations=self.n_simulations,
                               current_node=initial_node,
                               janggi_net=self.predictor,
                               temperature_start=1,
                               temperature_threshold=30,
                               temperature_end=0.01)
        player_red = NNPlayer(Color.RED,
                              n_simulations=self.n_simulations,
                              current_node=initial_node,
                              janggi_net=self.predictor,
                              temperature_start=1,
                              temperature_threshold=30,
                              temperature_end=0.01)
        game = Game(player_blue, player_red, board)
        while not game.is_finished(self.iter_max):
            new_action = game.get_next_action()
            game.actions.append(new_action)
            if game.current_player == Color.BLUE:
                examples.append([board.get_features(game.current_player, game.round),
                                 player_blue.current_node.get_policy(),
                                 None])
            else:
                examples.append([board.get_features(game.current_player, game.round),
                                 player_red.current_node.get_policy(),
                                 None])
            game.board.apply_action(new_action)
            game.switch_player()
            game.board.invalidate_action_cache(new_action)  # Try to reduce memory usage
            game.round += 1
        if not game.board.is_finished(game.current_player):
            score_BLUE = game.board.get_score(Color.BLUE)
            score_RED = game.board.get_score(Color.RED)
            if score_BLUE > score_RED:
                winner = Color.BLUE
            else:
                winner = Color.RED
        else:
            winner = Color(-game.current_player.value)
        for i, example in enumerate(examples):
            if winner == Color.BLUE and i%2 == 0:
                example[2] = 1
            elif winner == Color.RED and i%2 == 1:
                example[2] = 1
            else:
                example[2] = -1
        return examples

    def learn_policy(self, n_iterations, n_episodes):
        examples = []
        for _ in range(n_iterations):
            for ep in range(n_episodes):
                begin_time = time.time()
                examples += self.run_episode()
                print("Time Episode", ep, ": ", time.time() - begin_time)
            self.train(examples)
            player_red = RandomMCTSPlayer(Color.RED, n_simulations=self.n_simulations_opponent,
                                   temperature_start=0.01, temperature_threshold=30, temperature_end=0.01)
            player_blue = NNPlayer(Color.BLUE, n_simulations=self.n_simulations, janggi_net=self.predictor,
                                   temperature_start=0.01, temperature_threshold=30, temperature_end=0.01)
            fight(player_blue,
                  player_red,
                  self.iter_max)

    def train(self, examples):
        criterion = JanggiLoss()
        optimizer = torch.optim.SGD(self.predictor.parameters(), lr=0.02)
        random.shuffle(examples)

        for epoch in range(2):
            running_loss = 0.0
            for i, example in enumerate(examples):
                board, actions, value = example
                optimizer.zero_grad()
                policy, value_predicted = self.predictor(board)
                loss = criterion((policy, value_predicted), (actions, value))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i%100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
