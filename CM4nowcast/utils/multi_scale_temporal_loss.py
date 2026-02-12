import torch.nn as nn
import torch
import sys
sys.path.append("/root/code")
from pytorch_wavelets import DWTForward



class MTloss(nn.Module):
    def __init__(self, iterate = 3, scaler=0.02):
        super(MTloss, self).__init__()
        self.k_size = [5,7,11]
        self.iterate = iterate
        #self.loss_func = BMAELoss(scale=255.,mean=0.).cuda()
        self.loss_func = nn.L1Loss().cuda()
        self.scaler = scaler

    def forward(self, x, y):
        loss = []
        loss.append(self.loss_func(x[:,1:]-x[:,:11], y[:,1:]-y[:,:11])*self.scaler)
        for m in range(self.iterate):
            pf = nn.AvgPool2d(self.k_size[m], stride=1, padding=int((self.k_size[m] - 1) / 2)).cuda()
            loss.append(self.loss_func(pf(x[:,1:])-pf(x[:,:11]), pf(y[:,1:])-pf(y[:,:11]))*self.scaler)

        return loss


class MTloss_add_linear(nn.Module):
    def __init__(self, iterate = 3, scaler=0.02):
        super(MTloss_add_linear, self).__init__()
        self.k_size = [5, 7, 11, 12, 13, 14]
        self.iterate = iterate
        #self.loss_func = BMAELoss(scale=255.,mean=0.).cuda()
        self.loss_func = nn.L1Loss().cuda()
        self.scaler = scaler

    def forward(self, x, y):
        loss = 0
        loss += self.loss_func(x[:,1:]-x[:,:11], y[:,1:]-y[:,:11])*self.scaler
        for m in range(self.iterate):
            pf = nn.AvgPool2d(self.k_size[m], stride=1, padding=0).cuda()
            #pf = nn.AvgPool2d(self.k_size[m], stride=1, padding=int((self.k_size[m] - 1) / 2)).cuda()
            loss += self.loss_func(pf(x[:,1:])-pf(x[:,:11]), pf(y[:,1:])-pf(y[:,:11]))*self.scaler

        return loss



class MTloss_add_linear_adhoc(nn.Module):
    def __init__(self, iterate = 3, scaler=0.02):
        super(MTloss_add_linear_adhoc, self).__init__()
        self.k_size = [5,7,11]
        self.iterate = iterate
        #self.loss_func = BMAELoss(scale=255.,mean=0.).cuda()
        self.loss_func = nn.L1Loss().cuda()
        self.scaler = scaler
        self.weight = [3,5,7]

    def forward(self, x, y):
        loss = 0
        loss += self.loss_func(x[:,1:]-x[:,:11], y[:,1:]-y[:,:11])*self.scaler
        for m in range(self.iterate):
            pf = nn.AvgPool2d(self.k_size[m], stride=1, padding=int((self.k_size[m] - 1) / 2)).cuda()
            loss += self.loss_func(pf(x[:,1:])-pf(x[:,:11]), pf(y[:,1:])-pf(y[:,:11]))*self.scaler*self.weight[m]

        return loss

class WTloss(nn.Module):
    def __init__(self):
        super(WTloss, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = 3
        self.loss_func = nn.L1Loss().cuda()

    def forward(self, x, y):
        loss = 0
        loss = loss + self.loss_func(x[:,1:]-x[:,:11], y[:,1:]-y[:,:11])
        l, m, h = 1, 1, 1
        for i in range(self.iterate):
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss = loss + self.loss_func(x1[0][:,:,0][:,1:]-x1[0][:,:,0][:,:11], y1[0][:,:,0][:,1:]-y1[0][:,:,0][:,:11]) * m + self.loss_func(x1[0][:,:,1][:,1:]-x1[0][:,:,1][:,:11], y1[0][:,:,1][:,1:]-y1[0][:,:,1][:,:11]) * m + self.loss_func(x1[0][:,:,2][:,1:]-x1[0][:,:,2][:,:11], y1[0][:,:,2][:,1:]-y1[0][:,:,2][:,:11]) * h + self.loss_func(x0[:,1:]-x0[:,:11], y0[:,1:]-y0[:,:11]) * l
            x, y = x0, y0
        return loss


class WTlossM(nn.Module):
    def __init__(self, iterate = 3):
        super(WTlossM, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = iterate
        self.loss_func = nn.L1Loss().cuda()

    def forward(self, x, y):
        loss = []
        loss.append(self.loss_func(x[:,1:]-x[:,:11], y[:,1:]-y[:,:11]))
        for i in range(self.iterate):
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss.append(self.loss_func(x1[0][:,:,0][:,1:]-x1[0][:,:,0][:,:11], y1[0][:,:,0][:,1:]-y1[0][:,:,0][:,:11]))
            loss.append(self.loss_func(x1[0][:,:,1][:,1:]-x1[0][:,:,1][:,:11], y1[0][:,:,1][:,1:]-y1[0][:,:,1][:,:11]))
            loss.append(self.loss_func(x1[0][:,:,2][:,1:]-x1[0][:,:,2][:,:11], y1[0][:,:,2][:,1:]-y1[0][:,:,2][:,:11]))
            loss.append(self.loss_func(x0[:,1:]-x0[:,:11], y0[:,1:]-y0[:,:11]))
            x, y = x0, y0
        #loss.append(ssim_loss(x0, y0))
        return loss
