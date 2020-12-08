import json
from os import listdir

from janggi.game import Game
from janggi.player import RandomPlayer
from janggi.utils import Color

DIR = "pychess_games/"

if __name__ == "__main__":
    res = []
    for filename in listdir(DIR):
        with open(DIR + filename) as f:
            data = json.load(f)
        uci_usi = data["uci_usi"]
        if not uci_usi:
            continue
        try:
            game = Game.from_uci_usi(RandomPlayer(Color.BLUE),
                                     RandomPlayer(Color.RED),
                                     uci_usi)
        except:
            print("Problem with", filename)
            print(data["uci_usi"])
            print(data['result'])
            continue
        if data["result"] == "1-0":
            winner = Color.BLUE
        elif data["result"] == "0-1":
            winner = Color.RED
        else:
            print("DRAW")
            continue
        if winner == game.current_player:
            game.apply_action(None)
        res.append(game.dumps())
    with open("game_data2.txt", "w") as f:
        f.write("\n".join(res) + "\n")
