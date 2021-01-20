import os
import pickle
import random
import time

import torch

from janggi.parameters import BASE_ROOT_FILES


BASE_DIR = BASE_ROOT_FILES + "/inference/"
NEW_DIR = BASE_DIR + "new/"
OLD_DIR = BASE_DIR + "old/"


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
        time.sleep(0.01)
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
