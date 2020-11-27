import time
import logging

import numpy as np
from multiprocessing import shared_memory, Process
from multiprocessing import current_process

import torch
import multiprocessing as mp

from ia.janggi_network import JanggiNetwork
from ia.trainer import ModelSaver, run_episode_independant
from janggi.utils import BOARD_HEIGHT, BOARD_WIDTH, DEVICE

# Activate some logs, can be turned off
logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

# Number of residual layers in the network
N_RESIDUAL = 20

# Number of parallel simulations
N_POOLS = 25
# Number of simulation per round
N_SIMULATIONS = 800
# Number of round per game
ITER_MAX = 200
# Number of episodes between two saving and model reload
N_EPISODES = N_POOLS


# Parameters for parallel processing
# Maximum batch size processed at once
BATCH_SIZE = 16
# Number of features for the board
N_FEATURES = 16
# Number of features to represent a policy
N_FEATURES_POLICY = 58
# Number of buffers, where waiting data can be stored
N_BUFFERS = 2

DIMS = (BATCH_SIZE * N_BUFFERS * 2 + 1 + 1,  # 2 reading buffers, 2 writing buffers, 1 Info, 1 values
        N_FEATURES_POLICY,
        BOARD_HEIGHT,
        BOARD_WIDTH)

CURRENT_INDEX = (0, 0, 0, 0)


def create_shared_block():
    block = np.zeros(DIMS, dtype=np.float32)
    block[CURRENT_INDEX] = 0
    shm = shared_memory.SharedMemory(create=True, size=block.nbytes)
    shm_block = np.ndarray(block.shape, dtype=np.float32, buffer=shm.buf)
    shm_block[:] = block[:]
    return shm


def send_tensor(shr_name, tensor, lock):
    existing_shm = shared_memory.SharedMemory(name=shr_name)
    np_array = np.ndarray(DIMS, dtype=np.float32, buffer=existing_shm.buf)
    while True:
        with lock:
            current_index = int(np_array[CURRENT_INDEX])
            free_status = np_array[0, current_index + 1, 0, 1]
            if free_status < 0.1:
                np_array[CURRENT_INDEX] = (current_index + 1) % (BATCH_SIZE * N_BUFFERS)
                np_array[1 + current_index, :N_FEATURES, :, :] = tensor.numpy()
                np_array[0, current_index + 1, 0, 1] = 1
                break
    existing_shm.close()
    return current_index


def read_result(shr_name, current_index, lock):
    existing_shm = shared_memory.SharedMemory(name=shr_name)
    np_array = np.ndarray(DIMS, dtype=np.float32, buffer=existing_shm.buf)
    while True:
        with lock:
            if np_array[0, current_index + 1, 0, 0] > 0.1:
                policy = torch.tensor(np_array[1 + N_BUFFERS * BATCH_SIZE + current_index:2 + N_BUFFERS * BATCH_SIZE + current_index])
                value = torch.tensor(np_array[-1, current_index:current_index + 1, 0:1, 0])
                np_array[0, current_index + 1, 0, 0] = 0
                np_array[0, current_index + 1, 0, 1] = 0
                break
    existing_shm.close()
    return policy, value


def get_policy_value(model, features):
    features = features.to(DEVICE)
    with torch.no_grad():
        policy, value = model(features)
    return policy, value


def write_result(shr_name, previous_index, model, lock):
    existing_shm = shared_memory.SharedMemory(name=shr_name)
    np_array = np.ndarray(DIMS, dtype=np.float32, buffer=existing_shm.buf)
    while True:
        with lock:
            current_index = int(np_array[CURRENT_INDEX])
            if current_index > previous_index:
                features = np_array[1 + previous_index: 1 + current_index, :N_FEATURES, :, :]
                break
            if current_index < previous_index:
                features = np_array[np.r_[1: 1 + current_index,
                                          1 + previous_index: 1 + BATCH_SIZE * N_BUFFERS], :N_FEATURES, :, :]
                break
    features = torch.tensor(features)
    policy, value = get_policy_value(model, features)
    with lock:
        if current_index > previous_index:
            np_array[1 + previous_index + N_BUFFERS * BATCH_SIZE: 1 + current_index + N_BUFFERS * BATCH_SIZE, :, :, :] = policy.cpu()
            np_array[-1, previous_index:current_index, 0, 0] = value.view(-1).cpu()
            np_array[0, previous_index + 1:current_index + 1, 0, 0] = 1.0
        if current_index < previous_index:
            np_array[np.r_[1 + N_BUFFERS * BATCH_SIZE: 1 + current_index + N_BUFFERS * BATCH_SIZE,
                           1 + previous_index + N_BUFFERS * BATCH_SIZE: 1 + 2 * BATCH_SIZE * N_BUFFERS],
                     :N_FEATURES_POLICY, :, :] = policy.cpu()
            np_array[-1, np.r_[0: current_index, previous_index:BATCH_SIZE * N_BUFFERS], 0, 0] = value.view(-1).cpu()
            np_array[0, np.r_[1:current_index + 1, previous_index + 1:1 + BATCH_SIZE * N_BUFFERS], 0, 0] = 1.0
    existing_shm.close()
    return current_index


def get_model():
    model = JanggiNetwork(N_RESIDUAL)

    def load_latest_model():
        model_saver_temp = ModelSaver()
        model_saver_temp.load_latest_model(model)

    load_latest_model()
    model.to(DEVICE)
    model.eval()
    return model


def predictor_loop(shr_name, lock):
    current_index = 0
    model = get_model()
    while True:
        current_index = write_result(shr_name, current_index, model, lock)


class ProcessPredictor:

    def __init__(self, shr_name, lock):
        self.shr_name = shr_name
        self.lock = lock

    def __call__(self, features):
        current_index = send_tensor(self.shr_name, features, self.lock)
        result = read_result(self.shr_name, current_index, self.lock)
        return result


if __name__ == "__main__":
    if current_process().name == "MainProcess":
        mp.set_start_method("spawn", force=True)
        while True:
            with mp.Manager() as manager:
                print("Creating shared block")
                shr = create_shared_block()
                lock = manager.Lock()
                predictor = ProcessPredictor(shr.name, lock)
                model_saver = ModelSaver()

                print("Start Predictor Process")
                predictor_process = Process(target=predictor_loop, args=(shr.name, lock))
                predictor_process.start()
                time.sleep(5)

                begin_time = time.time()
                with mp.Pool(N_POOLS) as pool:
                    episodes = pool.map(run_episode_independant,
                                        [(predictor, N_SIMULATIONS, ITER_MAX) for _ in range(N_EPISODES)])
                examples = [x for episode in episodes for x in episode]
                model_saver.save_episodes(examples)
                print("Total time:", time.time() - begin_time)

                predictor_process.terminate()
                predictor_process.join()
                predictor_process.close()
                shr.close()
                shr.unlink()
