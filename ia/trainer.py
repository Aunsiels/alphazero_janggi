import copy
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

from multiprocessing import current_process

# Learning rate of the optimizer
LEARNING_RATE = 0.001

# Number of epoch when learning
EPOCH_NUMBER = 1
# Number of epoch when learning continuously
EPOCH_NUMBER_CONTINUOUS = 1
# When there is no episode to process, just wait
WAINTING_TIME_IF_NO_EPISODE = 1000

# For check if a model is better than the previous one, we perform some fights
N_FIGHTS = 100
# If the new model wins more than a certain percentage of games, we update the current model
VICTORY_THRESHOLD = 55

# When learning with existing data, how many game do we consider at once (
SUPERVISED_GAMES_FREQ = 30000

# During training, how often do we print loss
LOG_PRINT_FREQ = 1000

# For training
BATCH_SIZE = 16


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
        print("Setting trainer")
        self.predictor = predictor.to(DEVICE)
        self.n_simulations = n_simulations
        self.iter_max = iter_max
        self.n_simulations_opponent = n_simulation_opponent
        self.model_saver = ModelSaver(dir_base)
        self.optimizer = torch.optim.SGD(self.predictor.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
        self.model_saver.load_latest_model(self.predictor, self.optimizer)

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
        winner = game.get_winner()
        set_winner(examples, winner)
        return examples

    def learn_policy(self, n_iterations, n_episodes):
        for _ in range(n_iterations):
            if self.model_saver.has_last_episode():
                examples = self.model_saver.load_last_episode()
            else:
                examples = []
                for ep in range(n_episodes):
                    begin_time = time.time()
                    examples += self.run_episode()
                    print("Time Episode", ep, ": ", time.time() - begin_time)
                self.model_saver.save_episodes(examples)
            self.train_and_fight(examples)

    def learn_supervised(self, training_file):
        print("Generate training data...")
        with open(training_file) as f:
            examples_all = self._raw_to_examples(f)
        print("Start training")
        self.train_and_fight(examples_all)

    def _raw_to_examples(self, line_iterator):
        game_number = 1
        examples_all = []
        blue_starting = None
        red_starting = None
        board = None
        is_blue = True
        round = 0
        examples = []
        for line in line_iterator:
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
        return examples_all

    def continuous_learning(self):
        while True:
            if self.model_saver.has_last_episode_raw():
                print("Start new learning")
                self.continuous_learning_once()
            else:
                print("Waiting for more episodes")
                time.sleep(WAINTING_TIME_IF_NO_EPISODE)

    def continuous_learning_once(self):
        # First, train
        for _ in range(EPOCH_NUMBER_CONTINUOUS):
            all_examples = self._raw_to_examples(self.model_saver.all_episodes_raw_iterators())
            self.train(all_examples)
        # Then, fight!
        old_model = copy.deepcopy(self.predictor)
        self.model_saver.load_latest_model(old_model, None)
        old_model.to(DEVICE)
        victories = 0
        print("Start the fights!")
        for i in range(N_FIGHTS):
            new_player = NNPlayer(Color.BLUE,
                                  n_simulations=self.n_simulations,
                                  janggi_net=self.predictor,
                                  temperature_start=0.01,
                                  temperature_threshold=30,
                                  temperature_end=0.01)
            old_player = NNPlayer(Color.BLUE,
                                  n_simulations=self.n_simulations,
                                  janggi_net=old_model,
                                  temperature_start=0.01,
                                  temperature_threshold=30,
                                  temperature_end=0.01)
            if i < N_FIGHTS / 2:
                winner = fight(new_player, old_player, self.iter_max)
                if winner == Color.BLUE:
                    victories += 1
            else:
                winner = fight(old_player, new_player, self.iter_max)
                if winner == Color.RED:
                    victories += 1
        victory_percentage = victories / N_FIGHTS * 100
        if victory_percentage > VICTORY_THRESHOLD:
            # Replace model
            print("The model was good enough", victory_percentage)
            self.model_saver.save_weights(self.predictor, optimizer=self.optimizer)
            self.model_saver.rename_all_raw_episodes()
        else:
            # We take back the old model
            print("The model was not good enough", victory_percentage)
            self.model_saver.load_latest_model(self.predictor, optimizer=self.optimizer)

    def train_and_fight(self, examples):
        self.train(examples)
        self.organize_fight()

        self.model_saver.save_weights(self.predictor, optimizer=self.optimizer)
        self.model_saver.rename_last_episode()

    def organize_fight(self):
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

    def train(self, examples):
        self.predictor.train()
        criterion = JanggiLoss()
        dataset = ExampleDataset(examples)
        if examples:
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=0)
        else:
            dataloader = examples

        for epoch in range(EPOCH_NUMBER):
            running_loss = 0.0
            for i, example in enumerate(dataloader):
                board, actions, value = example
                self.optimizer.zero_grad()
                board = board.to(DEVICE)
                policy, value_predicted = self.predictor(board)
                value_predicted = value_predicted.view(-1, 1)
                policy = policy.to(DEVICE)
                value_predicted = value_predicted.to(DEVICE)
                actions = actions.to(DEVICE)
                value = value.view(-1, 1).to(DEVICE)
                loss = criterion((policy, value_predicted), (actions, value))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % LOG_PRINT_FREQ == LOG_PRINT_FREQ - 1:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / LOG_PRINT_FREQ))
                    running_loss = 0.0
        self.predictor.eval()


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
        if not os.path.isdir(dir_base + "/episodes_done"):
            os.mkdir(dir_base + "/episodes_done")
        if not os.path.isdir(dir_base + "/episodes_raw"):
            os.mkdir(dir_base + "/episodes_raw")
        if not os.path.isdir(dir_base + "/episodes_raw_done"):
            os.mkdir(dir_base + "/episodes_raw_done")
        if not os.path.isdir(dir_base + "/weights"):
            os.mkdir(dir_base + "/weights")
        self.model_path = dir_base + "/"
        self.episode_path = dir_base + "/episodes/"
        self.episode_done_path = dir_base + "/episodes_done/"
        self.episode_raw_path = dir_base + "/episodes_raw/"
        self.episode_raw_done_path = dir_base + "/episodes_raw_done/"
        self.weights_path = dir_base + "/weights/"

    def get_last_episode_index(self):
        maxi = -1
        for filename in os.listdir(self.episode_path):
            maxi = max(maxi, int(filename[len("episode_"):]))
        return maxi

    def get_last_episode_raw_index(self):
        maxi = -1
        for filename in os.listdir(self.episode_raw_path):
            maxi = max(maxi, int(filename[len("episode_"):]))
        return maxi

    def get_last_episode_done(self):
        maxi = -1
        for filename in os.listdir(self.episode_done_path):
            maxi = max(maxi, int(filename[len("episode_"):]))
        return maxi

    def get_last_episode_raw_done(self):
        maxi = -1
        for filename in os.listdir(self.episode_raw_done_path):
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

    def save_episodes_raw(self, episodes):
        new_index = self.get_last_episode_raw_index() + 1
        with open(self.episode_raw_path + "episode_" + str(new_index), "w") as f:
            f.write("\n".join(episodes) + "\n")

    def save_weights(self, model, optimizer):
        new_index = self.get_last_weight_index() + 1
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, self.weights_path + "weights_" + str(new_index))

    def load_latest_model(self, model, optimizer=None):
        last_index = self.get_last_weight_index()
        if last_index == -1:
            return
        self.load_index_model(model, optimizer, last_index)
        print("Model loaded:", last_index)

    def load_index_model(self, model, optimizer=None, last_index=-1):
        checkpoint = torch.load(self.weights_path + "weights_" + str(last_index))
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()

    def load_random_model(self, model, optimizer=None):
        last_index = self.get_last_weight_index()
        if last_index == -1:
            return -1
        model_idx = random.randint(0, last_index)
        self.load_index_model(model, optimizer, model_idx)
        print("Model loaded")
        return model_idx

    def has_last_episode(self):
        return self.get_last_episode_index() != -1

    def has_last_episode_raw(self):
        return self.get_last_episode_raw_index() != -1

    def load_last_episode(self):
        last_index = self.get_last_episode_index()
        if last_index == -1:
            print("No episode to load.")
            return None
        print("Load previous episode")
        return torch.load(self.episode_path + "episode_" + str(last_index))

    def all_episodes_iterators(self):
        for filename in os.listdir(self.episode_path):
            yield torch.load(self.episode_path + filename)

    def all_episodes_raw_iterators(self):
        for filename in os.listdir(self.episode_raw_path):
            with open(self.episode_raw_path + filename) as f:
                for line in f:
                    yield line

    def rename_last_episode(self):
        last_index = self.get_last_episode_index()
        if last_index == -1:
            return
        self.rename_episode_by_index(last_index)

    def rename_episode_by_index(self, last_index):
        last_index_done = self.get_last_episode_done()
        if last_index_done == -1:
            last_index_done = 0
        else:
            last_index_done += 1
        os.rename(self.episode_path + "episode_" + str(last_index),
                  self.episode_done_path + "episode_" + str(last_index_done))

    def rename_episode_raw_by_index(self, last_index):
        last_index_done = self.get_last_episode_raw_done()
        if last_index_done == -1:
            last_index_done = 0
        else:
            last_index_done += 1
        os.rename(self.episode_raw_path + "episode_" + str(last_index),
                  self.episode_raw_done_path + "episode_" + str(last_index_done))

    def rename_all_episodes(self):
        last_index = self.get_last_episode_index()
        while last_index != -1:
            self.rename_episode_by_index(last_index)
            last_index = self.get_last_episode_index()

    def rename_all_raw_episodes(self):
        last_index = self.get_last_episode_raw_index()
        while last_index != -1:
            self.rename_episode_raw_by_index(last_index)
            last_index = self.get_last_episode_raw_index()


def run_episode(trainer):
    print("Starting episode")
    begin_time = time.time()
    examples = trainer.run_episode()
    print("Time Episode: ", time.time() - begin_time)
    return examples


def run_episode_independant(args):
    print("Starting episode", current_process().name)
    begin_time = time.time()
    predictor, n_simulations, iter_max = args
    examples = []
    start_blue = random.choice(["won", "sang", "yang", "gwee"])
    start_red = random.choice(["won", "sang", "yang", "gwee"])
    board = Board(start_blue=start_blue, start_red=start_red)
    initial_node = MCTSNode(is_initial=True)
    player_blue = NNPlayer(Color.BLUE, n_simulations=n_simulations,
                           current_node=initial_node,
                           janggi_net=predictor,
                           temperature_start=1,
                           temperature_threshold=30,
                           temperature_end=0.01)
    player_red = NNPlayer(Color.RED,
                          n_simulations=n_simulations,
                          current_node=initial_node,
                          janggi_net=predictor,
                          temperature_start=1,
                          temperature_threshold=30,
                          temperature_end=0.01)
    game = Game(player_blue, player_red, board)
    while not game.is_finished(iter_max):
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
    winner = game.get_winner()
    set_winner(examples, winner)
    print("Time Episode: ", time.time() - begin_time)
    return examples


def run_episode_raw(args):
    print("Starting episode", current_process().name)
    begin_time = time.time()
    predictor, n_simulations, iter_max = args
    start_blue = random.choice(["won", "sang", "yang", "gwee"])
    start_red = random.choice(["won", "sang", "yang", "gwee"])
    board = Board(start_blue=start_blue, start_red=start_red)
    initial_node = MCTSNode(is_initial=True)
    player_blue = NNPlayer(Color.BLUE, n_simulations=n_simulations,
                           current_node=initial_node,
                           janggi_net=predictor,
                           temperature_start=1,
                           temperature_threshold=30,
                           temperature_end=0.01)
    player_red = NNPlayer(Color.RED,
                          n_simulations=n_simulations,
                          current_node=initial_node,
                          janggi_net=predictor,
                          temperature_start=1,
                          temperature_threshold=30,
                          temperature_end=0.01)
    game = Game(player_blue, player_red, board)
    while not game.is_finished(iter_max):
        new_action = game.get_next_action()
        game.actions.append(new_action)
        game.board.apply_action(new_action)
        game.switch_player()
        game.board.invalidate_action_cache(new_action)  # Try to reduce memory usage
        game.round += 1
    print("Time Episode: ", time.time() - begin_time)
    return game.dumps()
