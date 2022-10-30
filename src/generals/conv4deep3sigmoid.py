import torch.nn as nn
import torch

class Conv4Deep3(nn.Module):
    def __init__(self, device):
        super(Conv4Deep3, self).__init__()
        self.network = nn.Sequential(
            MountainPadder(device),
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, input):
        batch_size = input.shape[0]
        output = self.network(input)
        shape = output.shape
        flattened_output = output.view(shape[0], shape[1] * shape[2] * shape[3])
        return flattened_output

class MountainPadder(nn.Module):
    def __init__(self, device):
        super(MountainPadder, self).__init__()
        self.device = device


    def forward(self, input):
        mountain_column = torch.Tensor([[[[1]],[[0]],[[0]],[[0]]]]).to(self.device)
        side_pads = mountain_column.repeat(input.shape[0], 1, input.shape[2], 1)
        vertical_pads = mountain_column.repeat(input.shape[0], 1, 1, input.shape[3] + 2)
        horizontally_padded = torch.cat((side_pads, input, side_pads), dim=3)
        vertically_padded = torch.cat((vertical_pads, horizontally_padded, vertical_pads), dim=2)
        return vertically_padded