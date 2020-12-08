import os
import re

# Data from http://www.janggidosa.co.kr/
# I downloaded all the files in the data page (by hand, not that many)

# .gib = worst format in the world, it do not parse everything
# My goal is to create a new file, easier to read
# Beware, the output format is not standard

to_initial_board = {
    '마상상마': "won",
    "상마마상": "yang",
    "상마상마": "gwee",
    "마상마상": "sang"
}

BLUE_PLAYER = "초차림"
RED_PLAYER = "한차림"


def get_blue_red_starting_pos(header):
    blue_start = None
    red_start = None
    for line in header:
        split_temp = line.strip()[1:-1].split()
        if not split_temp:
            continue
        key = split_temp[0]
        value = re.sub('"', "", " ".join(split_temp[1:]))
        if key == BLUE_PLAYER:
            blue_start = to_initial_board[value]
        elif key == RED_PLAYER:
            red_start = to_initial_board[value]
    return blue_start, red_start


def get_moves(moves_initial_raw):
    moves_raw = " ".join(moves_initial_raw).strip()
    moves_raw = re.sub("<.*?>", "", moves_raw)
    moves_raw = re.sub("\s+", " ", moves_raw)
    moves_raw = re.sub("[^0-9 ]", "", moves_raw).strip()
    moves_list = moves_raw.split(" ")
    moves_res = []
    for i in range(0, len(moves_list), 2):
        if i+1 >= len(moves_list):
            moves_res.append("XXXX")  # Pass
            continue
        move = moves_list[i + 1]
        if (i+1) >= len(moves_list) or not move:
            moves_res.append("XXXX")
        else:
            if len(move) != 4:
                return []
            x_from = (int(move[0]) - 1)%10
            y_from = (int(move[1]) - 1)%9
            x_to = (int(move[2]) - 1)%10
            y_to = (int(move[3]) - 1)%9
            moves_res.append(str(x_from) + str(y_from) + str(x_to) + str(y_to))
    return moves_res

count = 0
keys = set()
invalid = []

res = []

for root, dirs, files in os.walk("gibo_files/", topdown=False):
    for name in files:
        header = []
        moves = []
        is_move = False
        comment = False
        filename = os.path.join(root, name)
        with open(filename, "rb") as f:
            for line in f:
                if 0xff in line:
                    continue
                line = line.decode("cp949")
                if line == "\r\n":
                    count += 1
                if line[0] == "[":
                    keys.add(line.split()[0][1:])
                if line[0] == "{" and "}" in line:
                    continue
                if line[0] == "{":
                    comment = True
                    continue
                if "}" in line:
                    comment = False
                    continue
                if comment:
                    continue
                if line == "\r\n":
                    if not header:
                        continue
                    elif not moves:
                        is_move = True
                    else:
                        blue_start, red_start = get_blue_red_starting_pos(header)
                        if blue_start is None or red_start is None:
                            invalid.append(filename)
                        else:
                            moves_res = get_moves(moves)
                            if moves_res:
                                res.append(blue_start + "\n" + red_start + "\n" + "\n".join(moves_res) + "\n")
                            else:
                                invalid.append(filename)
                        header = []
                        moves = []
                        is_move = False
                elif is_move:
                    moves.append(line)
                else:
                    header.append(line)

with open("game_data.txt", "w") as f:
    f.write("\n".join(res) + "\n")
