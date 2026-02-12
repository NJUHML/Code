# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class LinearWeightLoss(nn.Module):
    def __init__(self, num=2, epoch_size = 50):
        super(LinearWeightLoss, self).__init__()
        self.lambda_weight = np.ones([num, epoch_size], dtype = np.float32)

    def forward(self, x, epoch_idx=10):
        loss_sum = 0
        for i in range(len(x[0])):
            loss_sum += x[0][i]*1.0
        
        return loss_sum, [x[0][k].detach().cpu().numpy() for k in range(len(x[0]))]

if __name__ == '__main__':
    awl = LinearWeightLoss(2)
    print(awl.lambda_weight[:,10])
