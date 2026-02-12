# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, epoch_size = 50):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.lambda_weight = np.ones([num, epoch_size], dtype = np.float32)
        self.residual = np.ones([num, epoch_size], dtype = np.float32)

    def forward(self, x, epoch_idx=0):
        loss_sum = 0
        for i in range(len(x[0])):
            # print(f"self.params: {self.params}")
            loss_sum += 0.5 / (self.params[i] ** 2) * x[0][i] + torch.log(1 + self.params[i] ** 2)
            self.lambda_weight[i,epoch_idx] += (0.5 / (self.params[i] ** 2)).detach().numpy()
            self.residual[i,epoch_idx] += (torch.log(1 + self.params[i] ** 2)).detach().numpy()

        return loss_sum, [x[0][k].detach().cpu().numpy() for k in range(len(x[0]))]

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    # print(awl.parameters())
