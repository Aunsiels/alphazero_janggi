import os
import pickle
import random
import time

import torch

from ia.utils import generate_games

BASE_DIR = "inference/"
NEW_DIR = "inference/new/"
OLD_DIR = "inference/old/"

WITH_POOL = False

N_POOLS = 4
N_SIMULATIONS = 10
ITER_MAX = 10
N_EPISODES = N_POOLS

if not os.path.isdir(BASE_DIR):
    os.mkdir(BASE_DIR)
if not os.path.isdir(NEW_DIR):
    os.mkdir(NEW_DIR)
if not os.path.isdir(OLD_DIR):
    os.mkdir(OLD_DIR)


class FilePredictor:

    def __call__(self, features):
        filename = '{:010.6f}'.format(time.time()) + '{:01.10f}'.format(random.random())
        with open(NEW_DIR + filename + ".tmp", "wb") as f:
            pickle.dump(features, f)
        os.rename(NEW_DIR + filename + ".tmp", NEW_DIR + filename)
        while True:
            if os.path.isfile(OLD_DIR + filename):
                try:
                    with open(OLD_DIR + filename, "rb") as f:
                        policy, value = pickle.load(f)
                    os.remove(OLD_DIR + filename)
                    policy = torch.unsqueeze(policy, dim=0)
                    value = torch.unsqueeze(value, dim=0)
                    return policy, value
                except PermissionError:
                    time.sleep(0.01)
                    continue


if __name__ == "__main__":
    predictor = FilePredictor()
    generate_games(predictor, N_SIMULATIONS, ITER_MAX, WITH_POOL, N_POOLS, N_EPISODES)
