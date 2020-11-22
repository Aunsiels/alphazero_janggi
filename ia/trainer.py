import os
import random
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from ia.janggi_network import JanggiLoss
from ia.mcts import MCTSNode
from ia.random_mcts_player import NNPlayer, fight, RandomMCTSPlayer
from janggi.action import Action, get_none_action_policy
from janggi.board import Board
from janggi.game import Game
from janggi.player import RandomPlayer
from janggi.utils import Color, DEVICE

import multiprocessing as mp

SUPERVISED_GAMES_FREQ = 1000

LOG_PRINT_FREQ = 1000

BATCH_SIZE = 5

ASYNCHRONOUS = False


def set_winner(examples, winner):
    for example in examples:
        if winner == Color.BLUE and example[2] == Color.BLUE:
            example[2] = 1
        elif winner == Color.RED and example[2] == Color.RED:
            example[2] = 1
        else:
            example[2] = -1


class Trainer:

    def __init__(self, predictor, n_simulations=800, iter_max=200, n_simulation_opponent=800, dir_base="model"):
        self.predictor = predictor.to(DEVICE)
        self.n_simulations = n_simulations
        self.iter_max = iter_max
        self.n_simulations_opponent = n_simulation_opponent
        self.model_saver = ModelSaver(dir_base)
        self.model_saver.load_latest_model(self.predictor)

    def run_episode(self):
        examples = []
        start_blue = random.choice(["won", "sang", "yang", "gwee"])
        start_red = random.choice(["won", "sang", "yang", "gwee"])
        board = Board(start_blue=start_blue, start_red=start_red)
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
                                 player_blue.current_node.get_policy(game.current_player),
                                 Color.BLUE])
                examples.append([board.get_features(game.current_player, game.round, data_augmentation=True),
                                 player_blue.current_node.get_policy(game.current_player, data_augmentation=True),
                                 Color.BLUE])
            else:
                examples.append([board.get_features(game.current_player, game.round, data_augmentation=True),
                                 player_red.current_node.get_policy(game.current_player, data_augmentation=True),
                                 Color.RED])
                examples.append([board.get_features(game.current_player, game.round),
                                 player_red.current_node.get_policy(game.current_player),
                                 Color.RED])
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
        set_winner(examples, winner)
        return examples

    def learn_policy(self, n_iterations, n_episodes):
        for _ in range(n_iterations):
            if self.model_saver.has_last_episode():
                examples = self.model_saver.load_last_episode()
            else:
                if ASYNCHRONOUS:
                    with mp.Pool(3) as pool:
                        episodes = pool.map(run_episode, [self] * n_episodes)
                    examples = [x for episode in episodes for x in episode]
                else:
                    examples = []
                    for ep in range(n_episodes):
                        begin_time = time.time()
                        examples += self.run_episode()
                        print("Time Episode", ep, ": ", time.time() - begin_time)
                self.model_saver.save_episodes(examples)
            self.train_and_fight(examples)

    def learn_supervised(self, training_file):
        examples_all = []
        print("Generate training data...")
        game_number = 1
        with open(training_file) as f:
            blue_starting = None
            red_starting = None
            board = None
            is_blue = True
            round = 0
            examples = []
            for line in f:
                line = line.strip()
                if line == "":
                    if is_blue:
                        winner = Color.RED
                    else:
                        winner = Color.BLUE
                    set_winner(examples, winner)
                    examples_all += examples
                    # End game
                    blue_starting = None
                    red_starting = None
                    board = None
                    is_blue = True
                    round = 0
                    examples = []
                    game_number += 1
                    if game_number % SUPERVISED_GAMES_FREQ == 0:
                        print("Start training")
                        self.train_and_fight(examples_all)
                        examples_all = []
                        print("Generate training data...")
                elif blue_starting is None:
                    blue_starting = line
                elif red_starting is None:
                    red_starting = line
                else:
                    if board is None:
                        board = Board(start_blue=blue_starting, start_red=red_starting)
                    if line == "XXXX":
                        action = None
                        get_policy = get_none_action_policy
                    else:
                        action = Action(int(line[0]), int(line[1]), int(line[2]), int(line[3]))
                        get_policy = action.get_policy
                    if is_blue:
                        examples.append([board.get_features(Color.BLUE, round),
                                         get_policy(Color.BLUE),
                                         Color.BLUE])
                        examples.append([board.get_features(Color.BLUE, round, data_augmentation=True),
                                         get_policy(Color.BLUE,
                                                    data_augmentation=True),
                                         Color.BLUE])
                    else:
                        examples.append([board.get_features(Color.RED, round, data_augmentation=True),
                                         get_policy(Color.RED,
                                                    data_augmentation=True),
                                         Color.RED])
                        examples.append([board.get_features(Color.RED, round),
                                         get_policy(Color.RED),
                                         Color.RED])
                    board.apply_action(action)
                    round += 1
                    is_blue = not is_blue
            if is_blue:
                winner = Color.RED
            else:
                winner = Color.BLUE
            set_winner(examples, winner)
            examples_all += examples
        print("Start training")
        self.train_and_fight(examples_all)

    def train_and_fight(self, examples):
        self.train(examples)
        player_red = RandomPlayer(Color.RED)
        player_blue = NNPlayer(Color.BLUE, n_simulations=self.n_simulations, janggi_net=self.predictor,
                               temperature_start=0.01, temperature_threshold=30, temperature_end=0.01)
        fight(player_blue,
              player_red,
              self.iter_max)

        player_red = RandomMCTSPlayer(Color.RED, n_simulations=self.n_simulations_opponent,
                                      temperature_start=0.01, temperature_threshold=30, temperature_end=0.01)
        player_blue = NNPlayer(Color.BLUE, n_simulations=self.n_simulations, janggi_net=self.predictor,
                               temperature_start=0.01, temperature_threshold=30, temperature_end=0.01)
        fight(player_blue,
              player_red,
              self.iter_max)

        self.model_saver.save_weights(self.predictor)
        self.model_saver.rename_last_episode()

    def train(self, examples):
        criterion = JanggiLoss()
        optimizer = torch.optim.SGD(self.predictor.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001)
        dataset = ExampleDataset(examples)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0)

        for epoch in range(2):
            running_loss = 0.0
            for i, example in enumerate(dataloader):
                board, actions, value = example
                optimizer.zero_grad()
                board = board.to(DEVICE)
                policy, value_predicted = self.predictor(board)
                value_predicted = value_predicted.view(-1)
                policy = policy.to(DEVICE)
                value_predicted = value_predicted.to(DEVICE)
                actions = actions.to(DEVICE)
                value = value.to(DEVICE)
                loss = criterion((policy, value_predicted), (actions, value))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % LOG_PRINT_FREQ == LOG_PRINT_FREQ - 1:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / LOG_PRINT_FREQ / BATCH_SIZE))
                    running_loss = 0.0


class ExampleDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        return self.examples[index]

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)


class ModelSaver:

    def __init__(self, dir_base="model"):
        if not os.path.isdir(dir_base):
            os.mkdir(dir_base)
        if not os.path.isdir(dir_base + "/episodes"):
            os.mkdir(dir_base + "/episodes")
        if not os.path.isdir(dir_base + "/weights"):
            os.mkdir(dir_base + "/weights")
        self.model_path = dir_base + "/"
        self.episode_path = dir_base + "/episodes/"
        self.weights_path = dir_base + "/weights/"

    def get_last_episode_index(self):
        maxi = -1
        for filename in os.listdir(self.episode_path):
            if "done" in filename:
                continue
            maxi = max(maxi, int(filename[len("episode_"):]))
        return maxi

    def get_last_weight_index(self):
        maxi = -1
        for filename in os.listdir(self.weights_path):
            maxi = max(maxi, int(filename[len("weights_"):]))
        return maxi

    def save_episodes(self, episodes):
        new_index = self.get_last_episode_index() + 1
        torch.save(episodes, self.episode_path + "episode_" + str(new_index))

    def save_weights(self, model):
        new_index = self.get_last_weight_index() + 1
        torch.save(model.state_dict(), self.weights_path + "weights_" + str(new_index))

    def load_latest_model(self, model):
        last_index = self.get_last_weight_index()
        if last_index == -1:
            return
        self.load_index_model(model, last_index)
        print("Model loaded")

    def load_index_model(self, model, last_index):
        model.load_state_dict(torch.load(self.weights_path + "weights_" + str(last_index)))
        model.eval()

    def load_random_model(self, model):
        last_index = self.get_last_weight_index()
        if last_index == -1:
            return -1
        model_idx = random.randint(0, last_index)
        self.load_index_model(model, model_idx)
        print("Model loaded")
        return model_idx

    def has_last_episode(self):
        return self.get_last_episode_index() != -1

    def load_last_episode(self):
        last_index = self.get_last_episode_index()
        if last_index == -1:
            print("No episode to load.")
            return None
        print("Load previous episode")
        return torch.load(self.episode_path + "episode_" + str(last_index))

    def rename_last_episode(self):
        last_index = self.get_last_episode_index()
        if last_index == -1:
            return
        os.rename(self.episode_path + "episode_" + str(last_index),
                  self.episode_path + "episode_done_" + str(last_index))


def run_episode(trainer):
    print("Starting episode")
    begin_time = time.time()
    examples = trainer.run_episode()
    print("Time Episode: ", time.time() - begin_time)
    return examples
