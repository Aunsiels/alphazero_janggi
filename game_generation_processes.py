import time

import numpy as np
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import current_process
import multiprocessing as mp

import torch

from ia.janggi_network import JanggiNetwork
from ia.trainer import ModelSaver, run_episode_independant
from janggi.utils import BOARD_HEIGHT, BOARD_WIDTH, DEVICE

N_POOLS = 20
N_SIMULATIONS = 400
ITER_MAX = 200
N_EPISODES = 20
N_ITER_EPISODE = 10

BATCH_SIZE = 16
N_FEATURES = 16
N_FEATURES_POLICY = 58
N_BUFFERS = 2

DIMS = (BATCH_SIZE * N_BUFFERS * 2 + 1 + 1,  # 2 reading buffers, 2 writing buffers, 1 Info, 1 values
        N_FEATURES_POLICY,
        BOARD_HEIGHT,
        BOARD_WIDTH)

CURRENT_INDEX = (0, 0, 0, 0)


lock = Lock()


def create_shared_block():
    block = np.zeros(DIMS, dtype=np.float32)
    block[CURRENT_INDEX] = 0
    shm = shared_memory.SharedMemory(create=True, size=block.nbytes)
    shm_block = np.ndarray(block.shape, dtype=np.float32, buffer=shm.buf)
    shm_block[:] = block[:]
    return shm


def send_tensor(shr_name, tensor):
    existing_shm = shared_memory.SharedMemory(name=shr_name)
    np_array = np.ndarray(DIMS, dtype=np.float32, buffer=existing_shm.buf)
    while True:
        lock.acquire()
        current_index = int(np_array[CURRENT_INDEX])
        free_status = np_array[0, current_index + 1, 0, 1]
        if free_status < 0.1:
            np_array[CURRENT_INDEX] = (current_index + 1) % (BATCH_SIZE * N_BUFFERS)
            np_array[1 + current_index, :N_FEATURES, :, :] = tensor.numpy()
            np_array[0, current_index + 1, 0, 1] = 1
            lock.release()
            break
        lock.release()
    existing_shm.close()
    return current_index


def read_result(shr_name, current_index):
    existing_shm = shared_memory.SharedMemory(name=shr_name)
    np_array = np.ndarray(DIMS, dtype=np.float32, buffer=existing_shm.buf)
    while True:
        lock.acquire()
        if np_array[0, current_index + 1, 0, 0] > 0.1:
            policy = torch.tensor(np_array[1 + N_BUFFERS * BATCH_SIZE + current_index:2 + N_BUFFERS * BATCH_SIZE + current_index])
            value = torch.tensor(np_array[-1, current_index:current_index + 1, 0:1, 0])
            np_array[0, current_index + 1, 0, 0] = 0
            np_array[0, current_index + 1, 0, 1] = 0
            lock.release()
            break
        lock.release()
    existing_shm.close()
    return policy, value


def get_policy_value(model, features):
    features = features.to(DEVICE)
    with torch.no_grad():
        policy, value = model(features)
    return policy, value


def write_result(shr_name, previous_index, model):
    existing_shm = shared_memory.SharedMemory(name=shr_name)
    np_array = np.ndarray(DIMS, dtype=np.float32, buffer=existing_shm.buf)
    while True:
        lock.acquire()
        current_index = int(np_array[CURRENT_INDEX])
        if current_index > previous_index:
            features = np_array[1 + previous_index: 1 + current_index, :N_FEATURES, :, :]
            lock.release()
            break
        if current_index < previous_index:
            features = np_array[np.r_[1: 1 + current_index,
                                      1 + previous_index: 1 + BATCH_SIZE * N_BUFFERS], :N_FEATURES, :, :]
            lock.release()
            break
        lock.release()
    features = torch.tensor(features)
    policy, value = get_policy_value(model, features)
    lock.acquire()
    if current_index > previous_index:
        np_array[1 + previous_index + N_BUFFERS * BATCH_SIZE: 1 + current_index + N_BUFFERS * BATCH_SIZE, :, :, :] = policy
        np_array[-1, previous_index:current_index, 0, 0] = value.view(-1)
        np_array[0, previous_index + 1:current_index + 1, 0, 0] = 1.0
    if current_index < previous_index:
        np_array[np.r_[1 + N_BUFFERS * BATCH_SIZE: 1 + current_index + N_BUFFERS * BATCH_SIZE,
                       1 + previous_index + N_BUFFERS * BATCH_SIZE: 1 + 2 * BATCH_SIZE * N_BUFFERS],
                 :N_FEATURES_POLICY, :, :] = policy
        np_array[-1, np.r_[0: current_index, previous_index:BATCH_SIZE * N_BUFFERS], 0, 0] = value.view(-1)
        np_array[0, np.r_[1:current_index + 1, previous_index + 1:1 + BATCH_SIZE * N_BUFFERS], 0, 0] = 1.0
    lock.release()
    existing_shm.close()
    return current_index


def get_model():
    model = JanggiNetwork(20)

    def load_latest_model():
        model_saver_temp = ModelSaver()
        model_saver_temp.load_latest_model(model)

    load_latest_model()
    model.to(DEVICE)
    model.eval()
    return model


def predictor_loop(shr_name):
    current_index = 0
    model = get_model()
    while True:
        current_index = write_result(shr_name, current_index, model)


class ProcessPredictor:

    def __init__(self, shr_name):
        self.shr_name = shr_name

    def __call__(self, features):
        current_index = send_tensor(self.shr_name, features)
        result = read_result(self.shr_name, current_index)
        return result


def init(lock_temp):
    global lock
    lock = lock_temp


if __name__ == "__main__":
    if current_process().name == "MainProcess":
        print("Creating shared block")
        shr = create_shared_block()
        predictor = ProcessPredictor(shr.name)
        model_saver = ModelSaver()

        print("Start Predictor Process")
        predictor_process = Process(target=predictor_loop, args=(shr.name,))
        predictor_process.start()

        for _ in range(N_ITER_EPISODE):
            begin_time = time.time()
            with mp.Pool(N_POOLS, initializer=init, initargs=(lock,)) as pool:
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
