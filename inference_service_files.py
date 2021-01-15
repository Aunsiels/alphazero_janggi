import os
import pickle
import threading
import time
from queue import Queue

import torch

from ia.janggi_network import JanggiNetwork
from ia.trainer import ModelSaver
from janggi.utils import DEVICE


MULTITHREADED = True


BASE_DIR = "inference/"
NEW_DIR = "inference/new/"
OLD_DIR = "inference/old/"

N_RESIDUAL = 2
BATCH_SIZE = 16

if not os.path.isdir(BASE_DIR):
    os.mkdir(BASE_DIR)
if not os.path.isdir(NEW_DIR):
    os.mkdir(NEW_DIR)
if not os.path.isdir(OLD_DIR):
    os.mkdir(OLD_DIR)


def get_policy_value(model, features):
    features = features.to(DEVICE)
    with torch.no_grad():
        policy, value = model(features)
    return policy, value


def get_next_batch():
    filenames = sorted(
        filter(lambda x: not x.endswith(".tmp"),
               os.listdir(NEW_DIR)))[:BATCH_SIZE]
    arrays = get_features_list(filenames)
    if filenames:
        features = torch.cat(arrays)
        return features, filenames
    else:
        return None, filenames


def get_features_list(filenames):
    arrays = []
    if len(filenames) == BATCH_SIZE:
        print("FULL")
    for filename in filenames:
        while True:
            try:
                with open(NEW_DIR + filename, "rb") as f:
                    arrays.append(pickle.load(f))
                os.remove(NEW_DIR + filename)
                break
            except PermissionError:
                continue
    return arrays


def save_results(policy, value, filenames):
    policy = policy.cpu()
    value = value.cpu()
    for p, v, filename in zip(policy, value, filenames):
        with open(OLD_DIR + filename + ".tmp", "wb") as f:
            pickle.dump((p, v), f)
        os.rename(OLD_DIR + filename + ".tmp", OLD_DIR + filename)


def get_model():
    model = JanggiNetwork(N_RESIDUAL)

    def load_latest_model():
        model_saver_temp = ModelSaver()
        model_saver_temp.load_latest_model(model)

    load_latest_model()
    model.to(DEVICE)
    model.eval()
    return model


INPUT_QUEUE = Queue()
OUTPUT_QUEUE = Queue()


def read_features_thread():
    print("Start features thread.")
    while True:
        filenames = sorted(
            filter(lambda x: not x.endswith(".tmp"),
                   os.listdir(NEW_DIR)))[:BATCH_SIZE]
        arrays = get_features_list(filenames)
        if not arrays:
            time.sleep(0.001)
            continue
        for features, filename in zip(arrays, filenames):
            INPUT_QUEUE.put((features, filename))


def prediction_thread():
    print("Start prediction thread.")
    model = get_model()
    begin_time = time.time()
    total_processed = 0
    while True:
        if INPUT_QUEUE.empty():
            time.sleep(0.001)
            continue
        arrays = []
        filenames = []
        while len(arrays) < BATCH_SIZE and not INPUT_QUEUE.empty():
            features, filename = INPUT_QUEUE.get()
            arrays.append(features)
            filenames.append(filename)
        features = torch.cat(arrays)

        total_processed += len(filenames)
        if time.time() - begin_time > 10:
            print(int(total_processed / (time.time() - begin_time)), "annotations per second", end="\r")
            begin_time = time.time()
            total_processed = 0

        policy, value = get_policy_value(model, features)
        OUTPUT_QUEUE.put((policy, value, filenames))


def save_thread():
    print("Start saving thread.")
    while True:
        if OUTPUT_QUEUE.empty():
            time.sleep(0.001)
            continue
        policy, value, filenames = OUTPUT_QUEUE.get()
        save_results(policy, value, filenames)


if __name__ == "__main__":
    if MULTITHREADED:
        loading_thread = threading.Thread(target=read_features_thread)
        processing_thread = threading.Thread(target=prediction_thread)
        saving_thread = threading.Thread(target=save_thread)
        print("START THREADS")
        loading_thread.start()
        processing_thread.start()
        saving_thread.start()
        print("STARTED")
        processing_thread.join()
    else:
        model = get_model()
        begin_time = time.time()
        total_processed = 0

        while True:
            features, filenames = get_next_batch()
            if not filenames:
                continue

            total_processed += len(filenames)
            if time.time() - begin_time > 10:
                print(int(total_processed / (time.time() - begin_time)), "annotations per second", end="\r")
                begin_time = time.time()
                total_processed = 0

            policy, value = get_policy_value(model, features)
            save_results(policy, value, filenames)
