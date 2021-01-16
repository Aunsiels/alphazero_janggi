import json
import pickle
import socket
import urllib.request

import torch

from ia.utils import generate_games


WITH_POOL = False
N_POOLS = 3
N_SIMULATIONS = 10
ITER_MAX = 10
N_EPISODES = 20
WEB_SERVICE = False


class ServicePredictorWeb:

    def __init__(self, hostname="127.0.0.1", port=5000):
        self.url = "http://" + hostname + ":" + str(port) + "/predict"

    def __call__(self, features):
        data = {"features": features.tolist()}
        req = urllib.request.Request(self.url)
        req.add_header('Content-Type', 'application/json; charset=utf-8')
        json_data = json.dumps(data)
        json_data_as_bytes = json_data.encode('utf-8')  # needs to be bytes
        req.add_header('Content-Length', len(json_data_as_bytes))
        response = urllib.request.urlopen(req, json_data_as_bytes)
        json_result = json.load(response)
        return torch.tensor(json_result["policy"]), torch.tensor(json_result["value"])


HEADERSIZE = 10


class ServicePredictorSocket:

    def __init__(self, hostname="127.0.0.1", port=5000):
        self.connexion_info = (hostname, port)

    def __call__(self, features):
        data = pickle.dumps(features)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(self.connexion_info)
        msg = bytes(f"{len(data):<{HEADERSIZE}}", 'utf-8') + data
        client.sendall(msg)
        from_server = b''
        new_msg = True
        msg_length = 0
        while True:
            data = client.recv(4096)
            if new_msg:
                msg_length = int(data[:HEADERSIZE])
                new_msg = False
            from_server += data
            if len(from_server) - HEADERSIZE == msg_length:
                break
        json_data = pickle.loads(from_server[HEADERSIZE:])
        client.close()
        return json_data["policy"], json_data["value"]


if __name__ == "__main__":
    if WEB_SERVICE:
        predictor = ServicePredictorWeb()
    else:
        predictor = ServicePredictorSocket()
    generate_games(predictor, N_SIMULATIONS, ITER_MAX, WITH_POOL, N_POOLS, N_EPISODES)
