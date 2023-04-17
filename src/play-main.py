from play.client import Client
import time
import os
from generals import simpleConv, extraSimpleConv, conv2deep, conv3deep2, conv4deep3
import torch
import random
import play.tile as tile

client = Client()
updates_generator = client.get_updates()

client.set_username(os.environ['USER_ID'], os.environ['USERNAME'])

game = os.environ['GAME_ID']
print(f'got game id {game}')

client.join_game(game)

time.sleep(1)

print("forcing start")

client.set_force_start(game)


def find_all_owned_territory_coords(grid):

    owned_territory = []

    for r in range(client._map.rows):
        # print()
        for c in range(client._map.cols):
            if grid[r][c].isSelf() and grid[r][c].army > 1:
                owned_territory.append((r, c))

    return owned_territory

def convert_map_grid_to_tensor(grid):
    grid_tensor = torch.zeros([1, 4, 15, 15])
    for i, row in enumerate(grid):
        for j, t in enumerate(row):
            if t.isSelf():
                grid_tensor[0][1][i][j] = t.army
            elif t.army > 0:
                grid_tensor[0][2][i][j] = t.army
            elif t.tile == tile.TILE_MOUNTAIN:
                grid_tensor[0][0][i][j] = 1
            if t.isGeneral:
                grid_tensor[0][3][i][j] = 1

    return grid_tensor

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

def output_tensor_to_action(action_tensor):
    index = torch.argmax(action_tensor.flatten())
    height = client._map.rows
    width = client._map.rows
    print(f'height, width: {height}, {width}, {index}')
    move = index // (height * width)
    single_mat = index % (height * width)
    y = single_mat // width
    x = single_mat % height
    return y.item(), x.item(), DIRECTIONS[move]




time.sleep(2)
# general = conv4deep3.Conv4Deep3("cpu")
# general.load_state_dict(torch.load('src/generals/weights/simple-conv4deep2-3-Q-learning.weights'))
for output in updates_generator:
    grid = client._map.grid
    territories = find_all_owned_territory_coords(grid)
    # grid_tensor = convert_map_grid_to_tensor(grid)
    # action_tensor = general(grid_tensor)
    # y1, x1, direction = output_tensor_to_action(action_tensor)
    if len(territories) > 0:
        y1, x1 = random.choice(territories)

        direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        x2 = x1
        y2 = y1
        if direction == "UP":
            y2 -= 1
        elif direction == "DOWN":
            y2 += 1
        elif direction == "RIGHT":
            x2 += 1
        elif direction == "LEFT":
            x2 -= 1
        client.attack(y1, x1, y2, x2)
