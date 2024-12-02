import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
import os
import sys
import copy
import math
import numpy as np


class MLPDecoder(nn.Module):
    def __init__(self, feat_dim, num_points):
        super().__init__()
        self.np = num_points
        # (B,2046,1024)
        self.conv1 = nn.Sequential(nn.Conv1d(feat_dim * 2, 256, kernel_size=1, bias=False))
        self.dp1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False))
        self.dp2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.Conv1d(128, 3, kernel_size=1, bias=False),
                                   nn.Tanh())

    def forward(self, x):
        # x.shape: (bs,1024)
        batch_size = x.shape[0]

        x = self.conv1(x)  # B x 256 x C   # LBR
        x = self.dp1(x)  # B x 256 x C
        x = self.conv2(x)  # B x 256 x C
        x = self.dp2(x)  # B x 256 x C
        x = self.conv3(x)  # B x 3 x C
        x = x.view(batch_size, -1, 3)
        return x


#  B,1,1024
class MLPDecoder_assembly(nn.Module):
    def __init__(self, feat_dim, num_points):
        super().__init__()
        self.np = num_points
        # (B,2046,1024)
        self.fc_layers = nn.Sequential(
            nn.Linear(feat_dim * 2 , num_points * 2),
            nn.BatchNorm1d(num_points * 2),
            nn.Tanh(), # 使用tanh激活函数替换LeakyReLU
            nn.Linear(num_points * 2, num_points * 3),

        )

    def forward(self, x):
        # x.shape: (bs,1,1024)
        batch_size = x.shape[0]
        f = self.fc_layers(x)
        f = f.reshape(batch_size,self.np,3)

        return f