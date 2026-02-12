import torch
import torch.nn as nn
import torch.nn.functional as F



class LossFunctions():

    def __init__(self):

        self.alpha = 0.8

        self.w_fact = torch.Tensor([1.0]).cuda()
        self.w_exponent = torch.Tensor([0.012]).cuda()

        self.data_range = 1.0

        self.zero = torch.Tensor([0]).cuda()
        self.one = torch.Tensor([1]).cuda()
        self.max = torch.Tensor([100]).cuda()
        self.threshold = torch.Tensor([10.0]).cuda()


    def mse(self, output, target):
        """ Mean Squared Error Loss """

        criterion = torch.nn.MSELoss()
        loss = criterion(output, target)
        return loss



    def weighted_mse(self, output, target):
        """ Weighted Mean Squared Error Loss """
        a=0.0001
        weights = torch.where(target>0,torch.maximum(torch.tensor(2),0.5*torch.exp(torch.Tensor(target*16.0)+torch.log(torch.tensor(a)))-torch.tensor(a)),torch.tensor(1))
        loss = (weights * (output - target) ** 2).mean()
        return loss

    def weighted_mae(self, output, target):
        """ Weighted Mean Squared Error Loss """
        norm =torch.Tensor([1000.0]).cuda()
        alpha=torch.Tensor([0.1]).cuda()
        a=0.0001
        weights = torch.where(target>0,5,1)
        loss =(weights * torch.abs(output - target)).mean()
        return loss


    def multi_diff(self, pred, target):
        weights = 0.2
        thresholds=torch.Tensor([50.0,20.0,10.0,5.0,1.0]).cuda()
        n_thresholds = len(thresholds)
        total_loss = 0.0
        # norm =torch.Tensor([1000.0]).cuda()
        a=0.0001
        preds = torch.exp(torch.Tensor(pred*16.0)+torch.log(torch.tensor(a)))-torch.tensor(a)
        tar = torch.exp(torch.Tensor(target*16.0)+torch.log(torch.tensor(a)))-torch.tensor(a)
        # preds = pred*1000.
        # tar = target*1000.
        for i in range(n_thresholds):
            a=0.0001
            thld=thresholds[i]    
            sigmoid_pred_thld = torch.sigmoid(preds - thld)
            sigmoid_thld_pred = torch.sigmoid(thld - preds)
            TP_diff = ((tar > thld).float() * sigmoid_pred_thld).sum()
            FP_diff = ((tar < thld).float() * sigmoid_pred_thld).sum()
            FN_diff = ((tar > thld).float() * sigmoid_thld_pred).sum()
            BIAS_diff = (TP_diff+FP_diff) / (TP_diff+FN_diff+1e-10) 

            sigmoid_pred_thld0 = torch.sigmoid(tar - thld)
            sigmoid_thld_pred0 = torch.sigmoid(thld - tar)
            TP_diff0 = ((tar > thld).float() * sigmoid_pred_thld0).sum()
            FP_diff0 = ((tar < thld).float() * sigmoid_pred_thld0).sum()
            FN_diff0 = ((tar > thld).float() * sigmoid_thld_pred0).sum()
            BIAS_diff0 = (TP_diff0+FP_diff0) / (TP_diff0+FN_diff0+1e-10) 

            TS_diff = TP_diff / (TP_diff + FP_diff + FN_diff + 1e-10)
            total_loss += weights*torch.abs(BIAS_diff-BIAS_diff0)+weights*(1-TS_diff)
        
        return total_loss
    