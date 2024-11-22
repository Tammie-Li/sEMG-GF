"""
@ File: sEMGNet.py
@ Author: Tammie Li
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class EMGNet(nn.Module):
    def __init__(self, num_classes, drop_out, time_point, channel, N_t, N_s, window_size):
        """
        Desccription: Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). 
                      EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. 
                      Journal of neural engineering, 15(5), 056013.
        Params: num_classes: the number of classification category
                drop_out: the ratio of dropped network parameters
                time_point: the length of the convlution kernel in temporal extractor 
                channel: the number of electrode channel
                N_t: the number of kernels in temporal extractor 
                N_s: the number of kernels in spatial extractor 
        """
        super(EMGNet, self).__init__()

        # time_point and N_s needs to be a odd number
        if time_point % 2 == 0 or N_s // 2 == 1: 
            raise ValueError('Param time_point and N_s needs to be a odd number')

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((time_point//2, time_point//2+1, 0, 0)),
            nn.Conv2d(1, N_t, (1, time_point), bias=False),
            nn.BatchNorm2d(N_t)
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(N_t, N_s, (channel, 1), groups=N_t, bias=False),
            nn.BatchNorm2d(N_s),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_out)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((N_s//2-1, N_s//2, 0, 0)),
            nn.Conv2d(N_s, N_s, (1, N_s), groups=N_s, bias=False),
            nn.Conv2d(N_s, N_s, (1, 1), bias=False),
            nn.BatchNorm2d(N_s), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop_out)
        )
        
        if window_size == 0.25:
            tmp = 64
        elif window_size == 0.5:
            tmp = 128
        elif window_size == 0.75:
            tmp = 176

        self.fc1 = nn.Linear(tmp, num_classes)

    
    def forward(self, x):
        # 补零操作 N->C=8->T=250  ----->  N->C=8->T=256
        x = F.pad(x, (0, 6))
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.block_1(x)

        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)        

        logits = self.fc1(x)
        probas = F.softmax(logits, dim=1)
        return probas
    

if __name__ == "__main__":
    data = torch.rand(32, 8, 250)
    model = EMGNet(num_classes=5, drop_out=0.4, time_point=9, channel=8, N_t=8, N_s=16)
    print(model(data))
