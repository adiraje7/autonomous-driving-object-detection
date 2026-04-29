import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)
