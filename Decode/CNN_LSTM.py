"""
@ File: sEMGNet.py
@ Author: Tammie Li
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, window_size):
        """
        Desccription: CNN_LSTM
        """
        super(CNN_LSTM, self).__init__()

        self.hidden_size = 64
        self.num_layers = 2
        
        # CNN部分
        self.conv1 = nn.Conv1d(8, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM部分
        self.lstm = nn.LSTM(128, 64, 2, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # CNN部分
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # 调整形状为 (batch_size, seq_length, features) 用于LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM部分
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        
        return out

    

