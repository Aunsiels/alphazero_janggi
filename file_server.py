import time
import os
import random

from flask import Flask, request

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


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = request.get_data()
        filename = '{:010.6f}'.format(time.time()) + '{:01.10f}'.format(random.random())
        with open(NEW_DIR + filename + ".tmp", "wb") as f:
            f.write(features)
        os.rename(NEW_DIR + filename + ".tmp", NEW_DIR + filename)
        while True:
            if os.path.isfile(OLD_DIR + filename):
                try:
                    with open(OLD_DIR + filename, "rb") as f:
                        result = f.read()
                    os.remove(OLD_DIR + filename)
                    return result
                except PermissionError:
                    time.sleep(0.01)
                    continue
    return "404"


if __name__ == '__main__':
    app.run(threaded=True)