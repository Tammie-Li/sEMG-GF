"""
@ File: sEMGNet.py
@ Author: Tammie Li
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, num_classes, window_size):
        """
        Desccription: 1DCNN
        """
        super(LSTM, self).__init__()

        self.hidden_size = 64
        self.num_layers = 2
        
        # LSTM层
        self.lstm = nn.LSTM(8, self.hidden_size, 2, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        
        return out

    

