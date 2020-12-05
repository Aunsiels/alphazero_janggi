import pickle
import socket

import torch
from flask import Flask, jsonify, request

from ia.janggi_network import JanggiNetwork
from ia.trainer import ModelSaver
from janggi.utils import DEVICE

app = Flask(__name__)


def get_model():
    model = JanggiNetwork()

    def load_latest_model():
        model_saver = ModelSaver()
        model_saver.load_latest_model(model)

    load_latest_model()
    model.to(DEVICE)
    model.eval()
    return model


MODEL = get_model()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.json
        features = torch.tensor(json_data["features"], device=DEVICE)
        with torch.no_grad():
            policy, value = MODEL(features)
            res = jsonify({'policy': policy.tolist(), 'value': value.tolist()})
        return res
    return "404"


HEADERSIZE = 10


def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', 5000))
    server.listen(5)
    print("Server Started")
    while True:
        conn, addr = server.accept()
        from_client = b''
        new_msg = True
        msg_length = 0
        while True:
            data = conn.recv(4096)
            if new_msg:
                msg_length = int(data[:HEADERSIZE])
                new_msg = False
            from_client += data
            if len(from_client) - HEADERSIZE == msg_length:
                break
        features = pickle.loads(from_client[HEADERSIZE:]).to(DEVICE)
        with torch.no_grad():
            policy, value = MODEL(features)
            res = pickle.dumps({'policy': policy, 'value': value})
        msg = bytes(f"{len(res):<{HEADERSIZE}}", 'utf-8') + res
        conn.send(msg)
        conn.close()


if __name__ == '__main__':
    # app.run(threaded=False)
    run_server()
