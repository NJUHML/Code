import sys
sys.path.append('/root/code/pytorch_msssim')
import ssim
import torch.nn as nn
sys.path.append("/root/code")
from pytorch_wavelets import DWTForward


ssim_loss = ssim.SSIM(data_range=1.0, channel=12)

ssim_loss_11 = ssim.SSIM(data_range=1.0, channel=11)



class WSloss(nn.Module):
    def __init__(self, iterate = 3):
        super(WSloss, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = iterate
    def forward(self, x, y):
        loss = []
        loss.append(1 - ssim_loss(x, y))
        for i in range(self.iterate):
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss.append(1 - ssim_loss(x1[0][:,:,0], y1[0][:,:,0]))
            loss.append(1 - ssim_loss(x1[0][:,:,1], y1[0][:,:,1]))
            loss.append(1 - ssim_loss(x1[0][:,:,2], y1[0][:,:,2]))
            loss.append(1 - ssim_loss(x0, y0))
            x, y = x0, y0
        #loss.append(ssim_loss(x0, y0))
        return loss


class WSloss_linear_add(nn.Module):
    def __init__(self):
        super(WSloss_linear_add, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = 3
    def forward(self, x, y):
        loss = 13
        loss = loss - ssim_loss(x, y)
        l, m, h = 1, 1, 1
        for i in range(self.iterate):
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss = loss - ssim_loss(x1[0][:,:,0], y1[0][:,:,0]) * m - ssim_loss(x1[0][:,:,1], y1[0][:,:,1]) * m - ssim_loss(x1[0][:,:,2], y1[0][:,:,2]) * h - ssim_loss(x0, y0)
            x, y = x0, y0
        return loss


class WSloss_linear_add_adhoc(nn.Module):
    def __init__(self, multi_scaler = [0.25, 1., 4.]):
        super(WSloss_linear_add_adhoc, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = 3
        self.scaler = multi_scaler

    def forward(self, x, y):
        #loss = 13
        loss = 0
        loss = loss + 1 - ssim_loss(x, y)
        l = self.scaler[0]
        m = self.scaler[1]
        h = self.scaler[2]
        for i in range(self.iterate):
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss = loss + (1 - ssim_loss(x1[0][:,:,0], y1[0][:,:,0])) * m + (1- ssim_loss(x1[0][:,:,1], y1[0][:,:,1])) * m + (1 - ssim_loss(x1[0][:,:,2], y1[0][:,:,2])) * h + (1 - ssim_loss(x0, y0)) * l
            x, y = x0, y0
        return loss


class WSloss_init_0(nn.Module):
    def __init__(self):
        super(WSloss_init_0, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = 5
    def forward(self, x, y, r=0.7):
        loss = 0
        loss = loss - ssim_loss(x, y)
        l, m, h = 1, 1, 1
        for i in range(self.iterate):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss = loss - ssim_loss(x1[0][:,:,0], y1[0][:,:,0]) * m - ssim_loss(x1[0][:,:,1], y1[0][:,:,1]) * m - ssim_loss(x1[0][:,:,2], y1[0][:,:,2]) * h
            x, y = x0, y0
        loss = loss - ssim_loss(x0, y0) * l
        return loss





class WSloss_init_2(nn.Module):
    def __init__(self):
        super(WSloss_init_0, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = 5
    def forward(self, x, y, r=0.7):
        loss = 2
        loss = loss - ssim_loss(x, y)
        l, m, h = 1, 1, 1
        for i in range(self.iterate):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss = loss - ssim_loss(x1[0][:,:,0], y1[0][:,:,0]) * m - ssim_loss(x1[0][:,:,1], y1[0][:,:,1]) * m - ssim_loss(x1[0][:,:,2], y1[0][:,:,2]) * h
            x, y = x0, y0
        loss = loss - ssim_loss(x0, y0) * l
        return loss

class WSloss_init_0_11(nn.Module):
    def __init__(self):
        super(WSloss_init_0_11, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = 5
    def forward(self, x, y, r=0.7):
        loss = 0
        loss = loss - ssim_loss_11(x, y)
        l, m, h = 1, 1, 1
        for i in range(self.iterate):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss = loss - ssim_loss_11(x1[0][:,:,0], y1[0][:,:,0]) * m - ssim_loss_11(x1[0][:,:,1], y1[0][:,:,1]) * m - ssim_loss_11(x1[0][:,:,2], y1[0][:,:,2]) * h
            x, y = x0, y0
        loss = loss - ssim_loss_11(x0, y0) * l
        return loss



class WSloss_no_minus(nn.Module):
    def __init__(self):
        super(WSloss_no_minus, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = 5
    def forward(self, x, y, r=0.7):
        loss = 0
        loss = loss + ssim_loss(x, y)
        l, m, h = 1, 1, 1
        for i in range(self.iterate):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss = loss + ssim_loss(x1[0][:,:,0], y1[0][:,:,0]) * m + ssim_loss(x1[0][:,:,1], y1[0][:,:,1]) * m + ssim_loss(x1[0][:,:,2], y1[0][:,:,2]) * h
            x, y = x0, y0
        loss = loss + ssim_loss(x0, y0) * l
        return loss



class WSloss_Mean(nn.Module):
    def __init__(self, iterate = 5):
        super(WSloss_Mean, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = iterate
    def forward(self, x, y, r=0.7):
        loss = 0
        loss -= ssim_loss(x, y)
        l, m, h = 1, 1, 1
        for i in range(self.iterate):
            if i == 1:
                l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
                x0, x1 = self.dwt(x)
                y0, y1 = self.dwt(y)
            else:
                l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
                x0, x1 = self.dwt(x)
                y0, y1 = self.dwt(y)
                loss = loss - ssim_loss(x1[0][:,:,0], y1[0][:,:,0]) * m - ssim_loss(x1[0][:,:,1], y1[0][:,:,1]) * m - ssim_loss(x1[0][:,:,2], y1[0][:,:,2]) * h
            x, y = x0, y0
        loss -= ssim_loss(x0, y0) * l
        return loss
