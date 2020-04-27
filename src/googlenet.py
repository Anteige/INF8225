from inception import Inception
import torch
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, AdaptiveAvgPool2d

class GoogLeNet(Module):
    
    def __init__(self):
        super(GoogLeNet, self).__init__()
        
        self.conv1 = Conv2d(3, 64, bias=False, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(64, eps=0.001)
        self.maxpool1 = MaxPool2d(3, stride=2, ceil_mode=True)
        
        self.conv2 = Conv2d(64, 64, bias=False, kernel_size=1)
        self.bn2 = BatchNorm2d(64, eps=0.001)
        
        self.conv3 = Conv2d(64, 192, bias=False, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2d(192, eps=0.001)
        self.maxpool2 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.relu = ReLU(inplace=True)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(0.2)
        self.fc = Linear(1024, 120)
    
    def forward(self, x):       # N x 3 x 224 x 224
        
        x = self.conv1(x)       # N x 64 x 112 x 112
        x = self.bn1(x)         # N x 64 x 112 x 112
        x = self.relu(x)        # N x 64 x 112 x 112
        x = self.maxpool1(x)    # N x 64 x 56 x 56
        
        x = self.conv2(x)       # N x 64 x 56 x 56
        x = self.bn2(x)         # N x 64 x 56 x 56
        x = self.relu(x)        # N x 64 x 56 x 56

        x = self.conv3(x)       # N x 192 x 56 x 56
        x = self.bn3(x)         # N x 192 x 56 x 56
        x = self.relu(x)        # N x 192 x 56 x 56
        x = self.maxpool2(x)    # N x 192 x 28 x 28

        x = self.inception3a(x) # N x 256 x 28 x 28
        x = self.inception3b(x) # N x 480 x 28 x 28
        x = self.maxpool3(x)    # N x 480 x 14 x 14
        
        x = self.inception4a(x) # N x 512 x 14 x 14
        x = self.inception4b(x) # N x 512 x 14 x 14
        x = self.inception4c(x) # N x 512 x 14 x 14
        x = self.inception4d(x) # N x 528 x 14 x 14

        x = self.inception4e(x) # N x 832 x 14 x 14
        x = self.maxpool4(x)    # N x 832 x 7 x 7
        x = self.inception5a(x) # N x 832 x 7 x 7
        x = self.inception5b(x) # N x 1024 x 7 x 7

        x = self.avgpool(x)     # N x 1024 x 1 x 1
        x = torch.flatten(x, 1) # N x 1024
        x = self.dropout(x)
        x = self.fc(x)          # N x 120

        return x