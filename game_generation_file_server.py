import os
import pickle
import time
import multiprocessing as mp
import urllib.request

import torch

from ia.trainer import ModelSaver, run_episode_raw

BASE_DIR = "inference/"
NEW_DIR = "inference/new/"
OLD_DIR = "inference/old/"

N_POOLS = 4
N_SIMULATIONS = 100
ITER_MAX = 100
N_EPISODES = N_POOLS

if not os.path.isdir(BASE_DIR):
    os.mkdir(BASE_DIR)
if not os.path.isdir(NEW_DIR):
    os.mkdir(NEW_DIR)
if not os.path.isdir(OLD_DIR):
    os.mkdir(OLD_DIR)


HOSTNAME = "localhost"
PORT = 5000
URL = "http://" + HOSTNAME + ":" + str(PORT) + "/predict"


class FileServerPredictor:

    def __call__(self, features):
        features = pickle.dumps(features)
        req = urllib.request.Request(URL)
        req.add_header('Content-Length', len(features))
        response = urllib.request.urlopen(req, features)
        policy, value = pickle.loads(response.read())
        policy = torch.unsqueeze(policy, dim=0)
        value = torch.unsqueeze(value, dim=0)
        return policy, value


if __name__ == "__main__":
    model_saver = ModelSaver()
    predictor = FileServerPredictor()

    while True:
        begin_time = time.time()
        with mp.Pool(N_POOLS) as pool:
            episodes = pool.map(run_episode_raw,
                                [(predictor, N_SIMULATIONS, ITER_MAX) for _ in range(N_EPISODES)])
        model_saver.save_episodes_raw(episodes)
        print("Total time:", time.time() - begin_time)
