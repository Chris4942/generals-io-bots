import torch.nn as nn
import torch

class CityGeneral(nn.Module):
    def __init__(self, device, depth, hidden_channels=16):
        super(CityGeneral, self).__init__()
        middle_layer = []
        for layer in range(depth):
            middle_layer.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            middle_layer.append(nn.LeakyReLU())
        self.network = nn.Sequential(
            MountainPadder(device),
            nn.Conv2d(5, hidden_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            *middle_layer,
            nn.Conv2d(hidden_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        ).to(device)
        self.device = device
    
    def forward(self, input, print_stuff=False):
        batch_size = input.shape[0]
        output = self.network(input)
        shape = output.shape
        flattened_output = output.view(shape[0], shape[1] * shape[2] * shape[3]).to(self.device)
        if print_stuff:
            print(f'input device = {input.device}')
            print(f'output device = {output.device}')
            print(f'flattened output = {flattened_output.device}')
            print(f'self device = {self.device}')
        return flattened_output

class MountainPadder(nn.Module):
    def __init__(self, device):
        super(MountainPadder, self).__init__()
        self.device = device


    def forward(self, input):
        input = input.to(self.device)
        mountain_column = torch.Tensor([[[[1]],[[0]],[[0]],[[0]],[[0]]]]).to(self.device)
        side_pads = mountain_column.repeat(input.shape[0], 1, input.shape[2], 1)
        vertical_pads = mountain_column.repeat(input.shape[0], 1, 1, input.shape[3] + 2)
        horizontally_padded = torch.cat((side_pads, input, side_pads), dim=3)
        vertically_padded = torch.cat((vertical_pads, horizontally_padded, vertical_pads), dim=2)
        return vertically_padded