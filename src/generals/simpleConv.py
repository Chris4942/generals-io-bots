import torch.nn as nn

class SimpleConvGeneral(nn.Module):
    def __init__(self, height, width):
        super(SimpleConvGeneral, self).__init__()
        self.network = nn.Sequential(
            ConvChain(7, 16, 4),
            nn.Conv2d(16, 6, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
    
    def forward(self, input):
        return self.network(input)


class ConvChain(nn.Module):
    def __init__(self, input_channels, output_channels, depth):
        super(ConvChain, self).__init__()
        self.chain = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            *[
                nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1) 
                if i % 2 == 0 
                else nn.LeakyReLU() 
                for i in range (2 * (depth - 1))
            ]
        )
    
    def forward(self, input):
        return self.chain(input)