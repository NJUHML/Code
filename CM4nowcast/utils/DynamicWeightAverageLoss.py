# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class DynamicWeightAverageLoss(nn.Module):
    def __init__(self, num=2, epoch_size = 50, T = 2, batch=4):
        super(DynamicWeightAverageLoss, self).__init__()
        self.lambda_weight = np.ones([num, epoch_size], dtype = np.float32)
        self.avg_cost = np.zeros([epoch_size, num], dtype = np.float32)
        self.T = T
        self.num = num
        self.train_batch = batch
        self.cost = np.zeros(num, dtype=np.float32)

    def forward(self, x, epoch_idx=0):
        #for i in range(len(x[0])):
        #   self.cost[i] = x[0][i].item()
        #self.avg_cost[epoch_idx, :] += self.cost[:] / self.train_batch
        #assert len(x[0]) == self.num
        loss_sum = 0
        if epoch_idx < 2:
            #print('weight type 1')
            for i in range(len(x[0])):
                loss_sum += x[0][i]*1.0
        else:
            #print('weight type 2')
            w = []
            for i in range(len(x[0])):
                w.append(self.avg_cost[epoch_idx - 1, i] / self.avg_cost[epoch_idx - 2, i])
            wsum = 0
            for i in range(len(x[0])):
                wsum += np.exp(w[i] / self.T) 
            for i in range(len(x[0])):
                self.lambda_weight[i, epoch_idx] = self.num * np.exp(w[i] / self.T) / wsum
                loss_sum += self.lambda_weight[i, epoch_idx] * x[0][i]

        for i in range(len(x[0])):
           self.cost[i] = x[0][i].detach().cpu().numpy()
        self.avg_cost[epoch_idx, :] += self.cost[:] / self.train_batch
        
        return loss_sum, [x[0][k].detach().cpu().numpy() for k in range(len(x[0]))]

if __name__ == '__main__':
    awl = DynamicWeightAverageLoss(2)
    print(awl.lambda_weight())
