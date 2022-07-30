# The LNet architecture
# INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LeNet, self).__init__()

        # initialize the first block CONV=>RELU=>POOL
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels= 20, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # initialize the second block CONV=>RELU=>POOL
        self.conv2 = nn.Conv2d(in_channels=20, out_channels= 50, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # initialize the first FC layer FC => RELU
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        # initialize the softmax for the classifier
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)
        # softmax can be applied later when we predict the input
        self.logSoftmax = F.softmax(dim=1)

    def forward(self, t):
        # Pass the first block
        t = self.conv1(t)
        t = self.relu1(t)
        t = self.maxpool1(t)
        # Pass the second block
        t = self.conv2(t)
        t = self.relu2(t)
        t = self.maxpool2(t)
        # flatten the output
        # this step is improtant to remember the output dimension
        # before tensor: (batch_size, num_channels, height, width)
        # after flatten: (batch_size, 1-D values)
        t = torch.flatten(t, 1)
        t = self.fc1(t)
        t = self.relu3(t)
        # pass the output to the classifier
        t = self.fc2(t)
        out = self.logSoftmax(t)
        # return our predictions
        return out





        
    