import torch
from torch.nn import ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d

class Inception(Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = Sequential(
            Conv2d(in_channels, ch1x1, bias=False, kernel_size=1),
            BatchNorm2d(ch1x1, eps=0.001),
            ReLU(inplace=True)
        )
        
        self.branch2 = Sequential(
            Conv2d(in_channels, ch3x3red, bias=False, kernel_size=1),
            BatchNorm2d(ch3x3red, eps=0.001),
            ReLU(inplace=True),
            Conv2d(ch3x3red, ch3x3, bias=False, kernel_size=3, padding=1),
            BatchNorm2d(ch3x3, eps=0.001),
            ReLU(inplace=True)
        )

        self.branch3 = Sequential(
            Conv2d(in_channels, ch5x5red, bias=False, kernel_size=1),
            BatchNorm2d(ch5x5red, eps=0.001),
            ReLU(inplace=True),
            Conv2d(ch5x5red, ch5x5, bias=False, kernel_size=3, padding=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            BatchNorm2d(ch5x5, eps=0.001),
            ReLU(inplace=True)
        )

        self.branch4 = Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            Conv2d(in_channels, pool_proj, bias=False, kernel_size=1),
            BatchNorm2d(pool_proj, eps=0.001),
            ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
