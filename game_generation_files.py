import os
import pickle
import random
import time
import multiprocessing as mp

import torch

from ia.trainer import ModelSaver, run_episode_raw, run_episode_raw_loop

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


def save_queue_process(queue):
    model_saver = ModelSaver()
    begin_time = time.time()
    while True:
        if queue.qsize() < N_EPISODES:
            time.sleep(1)
            continue
        episodes = []
        for _ in range(N_EPISODES):
            episodes.append(queue.get())
        model_saver.save_episodes_raw(episodes)
        print("Total time:", time.time() - begin_time)
        begin_time = time.time()


if __name__ == "__main__":
    model_saver = ModelSaver()
    predictor = FilePredictor()

    while True:
        if WITH_POOL:
            begin_time = time.time()
            with mp.Pool(N_POOLS) as pool:
                episodes = pool.map(run_episode_raw,
                                    [(predictor, N_SIMULATIONS, ITER_MAX) for _ in range(N_EPISODES)])
            model_saver.save_episodes_raw(episodes)
            print("Total time:", time.time() - begin_time)
        else:
            OUTPUT_QUEUE = mp.Queue()
            processes = []
            saving_process = mp.Process(target=save_queue_process,
                                        args=(OUTPUT_QUEUE,))
            saving_process.start()
            for _ in range(N_POOLS):
                processes.append(mp.Process(target=run_episode_raw_loop,
                                            args=(predictor, N_SIMULATIONS, ITER_MAX, OUTPUT_QUEUE)))
            for process in processes:
                process.start()
            for process in processes:
                process.join()
