import json
import torch
from torch.utils.data import DataLoader, IterableDataset

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class JsonDataset(IterableDataset):
    def __init__(self, files):
        self.files = files
    
    def __iter__(self):
        for json_file in self.files:
            with open(json_file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    try:
                        x, y, direction = sample["moves"]['0']
                        target_tensor = torch.zeros(6, 15, 15)
                        if direction == UP:
                            target_tensor[0][y][x] = 1
                        elif direction == DOWN:
                            target_tensor[1][y][x] = 1
                        elif direction == LEFT:
                            target_tensor[2][y][x] = 1
                        elif direction == RIGHT:
                            target_tensor[3][y][x] = 1
                        map_array = sample["map"]
                        map_tensor = torch.Tensor(map_array).view(7, 15, 15)
                        yield map_tensor, target_tensor
                    except: 
                        pass

class ExtraSimplerDataset(IterableDataset):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for json_file in self.files:
            with open(json_file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    try:
                        target_tensor = sample["moves"]['0']
                    except:
                        continue
                    map_array = sample["map"]
                    map_tensor = torch.Tensor(map_array).view(7, 15, 15)[1:3]
                    yield map_tensor, target_tensor

class MountainDataset(IterableDataset):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for json_file in self.files:
            with open(json_file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    try:
                        target_tensor = sample["moves"]['0']
                    except:
                        continue
                    map_array = sample["map"]
                    map_tensor = torch.Tensor(map_array).view(7, 15, 15)[0:3]
                    yield map_tensor, target_tensor


class GeneralsAndMountainsDataset(IterableDataset):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for json_file in self.files:
            with open(json_file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    try:
                        target_tensor = sample["moves"]['0']
                    except:
                        continue
                    map_array = sample["map"]
                    map_tensor = torch.Tensor(map_array).view(7, 15, 15)[0:4]
                    yield map_tensor, target_tensor