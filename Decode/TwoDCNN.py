"""
@ File: sEMGNet.py
@ Author: Tammie Li
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoDCNN(nn.Module):
    def __init__(self, num_classes, window_size):
        """
        Desccription: 1DCNN
        """
        super(TwoDCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)

        if window_size == 0.25:
            t = 1152
        elif window_size == 0.5:
            t = 2176
        else:
            t = 3200

        self.fc1 = nn.Linear(t, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):

        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2, padding=1))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2, padding=1))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2, padding=1))
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)), 2, padding=1))
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x

    

