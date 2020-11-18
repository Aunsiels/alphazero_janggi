import json
import os
import random

from ia.janggi_network import JanggiNetwork
from ia.random_mcts_player import RandomMCTSPlayer, NNPlayer, fight
from ia.trainer import ModelSaver
from janggi.utils import Color


def get_player(player_name, color, model_saver):
    if player_name == "random_mcts":
        return RandomMCTSPlayer(color,
                                n_simulations=800,
                                temperature_start=0.01,
                                temperature_threshold=30,
                                temperature_end=0.01)
    else:
        predictor = JanggiNetwork()
        model_saver.load_index_model(predictor, player_name)
        return NNPlayer(color,
                        n_simulations=400,
                        janggi_net=predictor,
                        temperature_start=0.01,
                        temperature_threshold=30,
                        temperature_end=0.01)


def print_results(result):
    players = set()
    for blue, value in result.items():
        for red in value:
            players.add(red)
        players.add(blue)
    players = list(players)
    key_to_idx = {x: i for i, x in enumerate(players)}
    maxi_length = max([len(str(x)) for x in players])
    result_tab = [["X"  + " " * (maxi_length - 1) for _ in range(len(players) + 1)] for _ in range(len(players))]
    for i in range(len(players)):
        players[i] = str(players[i]) + " " * (maxi_length - len(str(players[i])))
        result_tab[i][0] = str(players[i])
    for blue, value_temp in result.items():
        for red, value in value_temp.items():
            result_tab[key_to_idx[blue]][key_to_idx[red] + 1] = str(int(100 * value[0] / (value[0] + value[1]))) + "% " + str(value[0]) + "/" + str(value[1])
            result_tab[key_to_idx[blue]][key_to_idx[red] + 1] = result_tab[key_to_idx[blue]][key_to_idx[red] + 1] + " " * \
                                                                (maxi_length - len(result_tab[key_to_idx[blue]][key_to_idx[red] + 1]))
    result_tab.insert(0, ["blue/red" + " " * (maxi_length - 8)] + [str(x) for x in players])
    print("\n".join(["\t".join(x) for x in result_tab]))


def save(result):
    with open("arena_temp.json", "w") as f:
        json.dump(result, f)


def load():
    if os.path.isfile("arena_temp.json"):
        print("Last arena loaded")
        with open("arena_temp.json") as f:
            res = json.load(f)
    else:
        print("Nothing to load.")
        res = dict()
    return res


if __name__ == "__main__":

    result = load()

    while True:

        players = ["random_mcts"]
        model_saver = ModelSaver()
        for i in range(model_saver.get_last_weight_index() + 1):
            players.append(i)
        if len(players) == 1:
            print("Not enough players")
            continue
        player_one_name, player_two_name = random.sample(players, k=2)
        player_one = get_player(player_one_name, Color.BLUE, model_saver)
        player_two = get_player(player_two_name, Color.RED, model_saver)

        player_one_name = str(player_one_name)
        player_two_name = str(player_two_name)

        if player_one_name not in result:
            result[player_one_name] = dict()
        if player_two_name not in result[player_one_name]:
            result[player_one_name][player_two_name] = [0, 0]

        winner = fight(player_one, player_two, 200)
        if winner == Color.BLUE:
            result[player_one_name][player_two_name][0] += 1
        else:
            result[player_one_name][player_two_name][1] += 1
        print_results(result)
        save(result)
