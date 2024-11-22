"""
@ File: sEMGNet.py
@ Author: Tammie Li
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class OneDCNN(nn.Module):
    def __init__(self, num_classes, window_size):
        """
        Desccription: 1DCNN
        """
        super(OneDCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        if window_size == 0.25:
            t = 992
        elif window_size == 0.5:
            t = 1984
        else:
            t = 2976

        self.fc1 = nn.Linear(t, 128)  # 动态调整输入维度
        self.fc2 = nn.Linear(128, num_classes)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    

