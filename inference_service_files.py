import os
import pickle

import torch

from ia.janggi_network import JanggiNetwork
from ia.trainer import ModelSaver
from janggi.utils import DEVICE

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
    arrays = []
    for filename in filenames:
        while True:
            try:
                with open(NEW_DIR + filename, "rb") as f:
                    arrays.append(pickle.load(f))
                os.remove(NEW_DIR + filename)
                break
            except PermissionError:
                continue
    if filenames:
        features = torch.cat(arrays)
        return features, filenames
    else:
        return None, filenames


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


if __name__ == "__main__":
    model = get_model()
    while True:
        features, filenames = get_next_batch()
        if not filenames:
            continue
        policy, value = get_policy_value(model, features)
        save_results(policy, value, filenames)
