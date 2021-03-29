import pickle
import urllib.request

import torch

from ia.utils import generate_games

WITH_POOL = False

N_POOLS = 64
N_SIMULATIONS = 800
ITER_MAX = 200
N_EPISODES = N_POOLS * 2


HOSTNAME = "gpu3"
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
    predictor = FileServerPredictor()

    generate_games(predictor, N_SIMULATIONS, ITER_MAX, WITH_POOL, N_POOLS, N_EPISODES)
