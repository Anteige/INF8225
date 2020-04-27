import torch
from torch.nn import ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Linear

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(3, 10, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(10),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(10, 20, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(20),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(62720, 120) 
        )
  
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x