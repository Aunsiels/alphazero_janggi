import json
import urllib.request
import urllib.error

from websocket import create_connection


def get_game_by_id(game_id):
    ws = create_connection("wss://www.pychess.org/wsr")
    ws.send('{"type":"board","gameId":"' + game_id + '"}')
    result = ws.recv()
    try:
        result_data = json.loads(result)
        if "gameId" in result_data:
            with open("pychess_games/" + result_data["gameId"] + ".json", "w") as f:
                f.write(result)
    except json.decoder.JSONDecodeError:
        print("Problem with", result)
    ws.close()


def get_game_ids():
    page_number = 1873
    while True:
        print(page_number)
        try:
            with urllib.request.urlopen("https://www.pychess.org/api/Fairy-Stockfish/perf/janggi?p=" +
                                        str(page_number)) as url:
                data = json.loads(url.read().decode())
                for game in data:
                    yield game["_id"]
            page_number += 1
        except urllib.error.HTTPError:
            break


if __name__ == "__main__":
    for game_id in get_game_ids():
        get_game_by_id(game_id)
