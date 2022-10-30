import json
import sys
import torch

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'


def convert_moves_to_machine_output(move):
    def convert_number_to_point(number):
        return number % 15, number // 15
    def get_my_format():
        x1, y1 = convert_number_to_point(move["start"])
        x2, y2 = convert_number_to_point(move["end"])
        x_diff, y_diff = x2 - x1 , y2 - y1
        if x_diff == 1:
            return x1, y1, RIGHT
        elif x_diff == -1:
            return x1, y1, LEFT
        elif y_diff == 1:
            return x1, y1, DOWN
        elif y_diff == -1:
            return x1, y1, UP
    x, y, direction = get_my_format()
    target_tensor = torch.zeros(4, 15, 15)
    if direction == UP:
        target_tensor[0][y][x] = 1
    elif direction == DOWN:
        target_tensor[1][y][x] = 1
    elif direction == LEFT:
        target_tensor[2][y][x] = 1
    elif direction == RIGHT:
        target_tensor[3][y][x] = 1
    target_tensor = target_tensor.view(1, 900)
    target_tensor = torch.argmax(target_tensor, dim = 1)
    return target_tensor[0].item()
    

def convert_map_to_machine_map(og_map, mountains, generals):
    generals = og_map["map"]["_map"]
    armies = og_map["map"]["_armies"]
    tensor = [[0] * len(generals)] * 7
    for i in range(len(generals)):
        if generals[i] == -1:
            continue
        if generals[i] == 0:
            tensor[1][i] = armies[i]
        else:
            tensor[2][i] = armies[i]
    
    for m in mountains:
        tensor[0][m] = 1
    for g in generals:
        tensor[3][g] = 1

    return tensor

if __name__ == "__main__":
    directory = sys.argv[1]

    moves_file = open(f'{directory}/moves', "r")
    move_lines = moves_file.readlines()

    try:
        mountain_file = open(f'{directory}/mountains', "r")
        mountains = json.loads(mountain_file.readline())
    except:
        print(f'unable to open mountains file in {directory}. Using empty mountains array instead!')
        mountains = []
    try:
        generals_file = open(f'{directory}/generals', "r")
        generals = json.loads(generals_file.readline())
    except:
        print(f'unable to open generals file. defaulting to blank file')
        generals = []


    moves = {}

    for line in move_lines:
        moves_array = json.loads(line)
        for move in moves_array:
            if move["turn"] not in moves:
                moves[move["turn"]] = {}
            moves[move["turn"]][move["index"]] = move

    map_file = open(f'{directory}/maps', "r")
    map_lines = map_file.readlines()

    maps = {}

    for line in map_lines:
        current_map = json.loads(line)
        maps[current_map["turn"]] = current_map

    total_turns = maps[len(maps.values()) - 1]["turn"]

    map_move_pairs = []

    for turn in range(total_turns):
        if turn in moves:
            map_move_pairs.append ({
                "moves": {k: convert_moves_to_machine_output(m) for k, m in moves[turn].items()},
                "map": convert_map_to_machine_map(maps[turn], mountains, generals),
            })

    output_file = open(f'{directory}/paired-data', 'w')
    json_output = [json.dumps(item) for item in map_move_pairs]
    for item in json_output:
        output_file.write(f'{item}\n')
