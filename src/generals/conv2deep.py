import torch.nn as nn

class Conv2Deep(nn.Module):
    def __init__(self):
        super(Conv2Deep, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
    
    def forward(self, input):
        batch_size = input.shape[0]
        return self.network(input).view(batch_size, 900)