import torch
import sys


class BMAELoss(torch.nn.Module):
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44):
        super(BMAELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
    def forward(self,y_pre,y_true):
        
        #assert y_true.min() >= 0
        #assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]
            
        return torch.mean(w_true * (abs(y_pre - y_true)))



class FCBMAELoss(torch.nn.Module):
    # Focal*Change_rate*Balance
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44, pre_len = 12):
        super(FCBMAELoss,self).__init__()
        self.pre_len = pre_len
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]

    def forward(self,y_pre,y_true,x_true_l,w_LK):
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]
        w_f = torch.pow((((torch.sum(abs(y_pre - y_true),(2,3))-torch.min(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1))/(torch.max(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1) - torch.min(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1)))+0.01),2).unsqueeze(2).unsqueeze(3)
        x_y = torch.cat((x_true_l, y_true),dim=1)
        w_c_v = (x_y[:,1:1+self.pre_len]-x_y[:,:self.pre_len])+(torch.max(y_true,1).values.unsqueeze(1)-torch.min(y_true,1).values.unsqueeze(1))
        return torch.mean(w_f * (w_c_v+torch.nn.functional.interpolate(w_LK,size=(384,384),mode='bilinear')) * w_true * (abs(y_pre - y_true)))

class FCBMSELoss(torch.nn.Module):
    # Focal*Change_rate*Balance
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44, pre_len = 12):
        super(FCBMSELoss,self).__init__()
        self.pre_len = pre_len
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]

    def forward(self,y_pre,y_true,x_true_l,w_LK):
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]
        w_f = torch.pow((((torch.sum(abs(y_pre - y_true),(2,3))-torch.min(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1))/(torch.max(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1) - torch.min(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1)))+0.01),2).unsqueeze(2).unsqueeze(3)
        x_y = torch.cat((x_true_l, y_true),dim=1)
        w_c_v = (x_y[:,1:1+self.pre_len]-x_y[:,:self.pre_len])+(torch.max(y_true,1).values.unsqueeze(1)-torch.min(y_true,1).values.unsqueeze(1))
        return torch.mean(w_f * (w_c_v+torch.nn.functional.interpolate(w_LK,size=(384,384),mode='bilinear')) * w_true * (y_pre - y_true)**2)


class CBMAELoss(torch.nn.Module):
    # Change_rate*Balance
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44, pre_len = 12):
        super(CBMAELoss,self).__init__()
        self.pre_len = pre_len
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]

    def forward(self,y_pre,y_true,x_true_l,w_LK):
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]
        x_y = torch.cat((x_true_l, y_true),dim=1)
        w_c_v = abs(x_y[:,1:1+self.pre_len]-x_y[:,0:1])
        w_c_v_1 = w_c_v.clone()
        for i in range(12):
            w_c_v_1[:,i]=torch.sum(w_c_v[:,:i],1)
        w_LK_1 = w_LK.clone()
        for i in range(12):
            w_LK_1[:,i]=torch.sum(w_LK[:,:i],1)
        return torch.mean((w_true+w_c_v_1+torch.nn.functional.interpolate(w_LK_1,size=(384,384),mode='bilinear')) * (abs(y_pre - y_true)))

class CBMSELoss(torch.nn.Module):
    # Change_rate*Balance
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44, pre_len = 12):
        super(CBMSELoss,self).__init__()
        self.pre_len = pre_len
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]

    def forward(self,y_pre,y_true,x_true_l,w_LK):
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]
        x_y = torch.cat((x_true_l, y_true),dim=1)
        w_c_v = abs(x_y[:,1:1+self.pre_len]-x_y[:,0:1])
        w_c_v_1 = w_c_v.clone()
        for i in range(12):
            w_c_v_1[:,i]=torch.sum(w_c_v[:,:i],1)
        w_LK_1 = w_LK.clone()
        for i in range(12):
            w_LK_1[:,i]=torch.sum(w_LK[:,:i],1)
        return torch.mean((w_true+w_c_v_1+torch.nn.functional.interpolate(w_LK_1,size=(384,384),mode='bilinear')) * (y_pre - y_true)**2)

class VBMAELoss(torch.nn.Module):
    # Change_rate*Balance
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44, pre_len = 12):
        super(VBMAELoss,self).__init__()
        self.pre_len = pre_len
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        self.ssim = SSIM()

    def forward(self,y_pre,y_true,x_true_l,w_LK):
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]
        x_y = torch.cat((x_true_l, y_true),dim=1)
        ssim1 = (self.ssim(y_true, x_true_l.repeat(1,12,1,1)))
        #CV = (((((y_true+1)/(y_true.mean((2,3)).unsqueeze(2).unsqueeze(3)+1))+1)**2).mean((2,3)).unsqueeze(2).unsqueeze(3))**(1/2)
        
        return torch.mean(ssim1*(w_true) * (abs(y_pre - y_true)))

class VBMSELoss(torch.nn.Module):
    # Change_rate*Balance
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44, pre_len = 12):
        super(VBMSELoss,self).__init__()
        self.pre_len = pre_len
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        self.ssim = SSIM()

    def forward(self,y_pre,y_true,x_true_l,w_LK):
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]
        x_y = torch.cat((x_true_l, y_true),dim=1)
        ssim1 = (self.ssim(y_true, x_true_l.repeat(1,12,1,1)))
        #CV = (((((y_true+1)/(y_true.mean((2,3)).unsqueeze(2).unsqueeze(3)))+1)**2).mean((2,3)).unsqueeze(2).unsqueeze(3))**(1/2)

        return torch.mean(ssim1 *(w_true) * ((y_pre - y_true)**2))


class FMSELoss(torch.nn.Module):
    def __init__(self):
        super(FMSELoss,self).__init__()

    def forward(self,y_pre,y_true):
        w_f = torch.pow((((torch.sum(abs(y_pre - y_true),(2,3))-torch.min(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1))/(torch.max(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1) - torch.min(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1)))+0.01),2)
        return torch.sum(torch.sum(w_f,1).unsqueeze(1)*torch.sum((y_pre - y_true)**2,(2,3)))*0.0001

class FMAELoss(torch.nn.Module):
    def __init__(self):
        super(FMAELoss,self).__init__()

    def forward(self,y_pre,y_true):
        w_f = torch.pow((((torch.sum(abs(y_pre - y_true),(2,3))-torch.min(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1))/(torch.max(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1) - torch.min(torch.sum(abs(y_pre - y_true),(2,3)),1).values.unsqueeze(1)))+0.01),2)
        return torch.sum(torch.sum(w_f,1).unsqueeze(1)*torch.sum(abs(y_pre - y_true),(2,3)))*0.0001



class BMSELoss(torch.nn.Module):
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44):
        super(BMSELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
    def forward(self,y_pre,y_true):
        
        #assert y_true.min() >= 0
        #assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i] #获取权重矩阵
            
        return torch.mean(w_true * (y_pre - y_true)**2)   

class MSELoss(torch.nn.Module):
    def __init__(self, weights = [1,2,5,10,30], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44):
        super(MSELoss,self).__init__()

        assert len(weights) == len(thresholds)
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]

    def forward(self,y_pre,y_true):

        return torch.mean((y_pre - y_true)**2)


class MSELoss_Dense(torch.nn.Module):
    def __init__(self, weights):
        super(MSELoss_Dense,self).__init__()
    
        self.weights = weights

    def forward(self,y_pre,y_true):

        return torch.mean(self.weight * (y_pre - y_true)**2)



class BMSELossb(torch.nn.Module):
    def __init__(self, weights = [1,5,30,60,120], thresholds = [31,74,133,181,255], scale=47.54,mean=33.44):
        super(BMSELossb,self).__init__()

        assert len(weights) == len(thresholds)
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds]
        #[0.25, 0.375, 0.5, 0.625, 1.0]

    def forward(self,y_pre,y_true):

        #assert y_true.min() >= 0
        #assert y_true.max() <= 1

        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i] #获取权重矩阵

        return torch.mean(w_true * (y_pre - y_true)**2)


 
class BMSAELoss(torch.nn.Module):
    def __init__(self, weights = [1,2,5,10,30], 
                 thresholds = [31,74,133,181,255],scale=47.54,mean=33.44,
                 mse_w = 1,mae_w = 1):
        super(BMSAELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        self.weights = weights
        self.thresholds = [(threshold-mean)/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        self.mse_w = mse_w
        self.mae_w = mae_w
        
    def forward(self,y_pre,y_true):
        
        #assert y_true.min() >= 0
        #assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i] #获取权重矩阵
            
        return self.mse_w*torch.mean(w_true * (y_pre - y_true)**2) + self.mae_w*torch.mean(w_true * (abs(y_pre - y_true)))
    
#%%
class STBMSELoss(torch.nn.Module):
    def __init__(self, spatial_weights = [1,2,5,10,30],
                 thresholds = [20,30,40,50,80],time_weight_gap = 1):
        super(STBMSELoss,self).__init__()
        
        assert len(spatial_weights) == len(thresholds)
        scale = max(thresholds)
        self.spatial_weights = spatial_weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
        self.time_weight_gap = time_weight_gap
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        sim = SSIM()_true: 4D or 5D Tensor
            real value.
        '''
        
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.spatial_weights)):
            w_true[w_true < self.thresholds[i]] = self.spatial_weights[i] #获取权重矩阵
        
        
        if len(y_true.size()) == 4:
            batch, seq, height, width = y_true.shape
            # y_true = np.expand_dims(y_true,axis = 2)
            # y_pre = np.expand_dims(y_pre,axis = 2)
            # w_true = np.expand_dims(w_true,axis = 2)
        
        if len(y_true.size()) == 5:
            batch,seq, channel,height,width = y_true.shape
            assert channel == 1
            
        time_weight = torch.arange(0,seq)*self.time_weight_gap + 1 
        time_weight = time_weight.to(y_pre.device)
        
        all_loss = 0
        for i in range(seq):
            loss = torch.mean(w_true[:,i]*(y_pre[:,i] - y_true[:,i])**2)
            all_loss += time_weight[i]*loss
        
        return all_loss
       


class MSEL1Loss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(MSEL1Loss, self).__init__()
        self.alpha = alpha

        self.mse_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()

    def __call__(self, input, target):
        mse_loss = self.mse_criterion(input, target)
        l1_loss = self.l1_criterion(input, target)
        loss = mse_loss + self.alpha * l1_loss
        return loss / 2.

class Dwt(torch.nn.Module):
    def __init__(self):
        super(Dwt, self).__init__()

        self.dwt1 = DWTForward(J=1, wave='haar', mode='symmetric')
        self.dwt2 = DWTForward(J=1, wave='haar', mode='symmetric')
        self.dwt3 = DWTForward(J=1, wave='haar', mode='symmetric')


    def forward(self, x):
        dwt1_1_l, dwt1_1_h = self.dwt1(x)
        dwt1_1 = torch.cat((dwt1_1_l, dwt1_1_h[0][:,:,0], dwt1_1_h[0][:,:,1], dwt1_1_h[0][:,:,2]), dim=1)
        dwt2_1_l, dwt2_1_h = self.dwt2(dwt1_1_l)
        dwt2_1 = torch.cat((dwt2_1_l, dwt2_1_h[0][:, :, 0], dwt2_1_h[0][:, :, 1], dwt2_1_h[0][:, :, 2]), dim=1)
        dwt3_1_l, dwt3_1_h = self.dwt3(dwt2_1_l)
        dwt3_1 = torch.cat((dwt3_1_l, dwt3_1_h[0][:, :, 0], dwt3_1_h[0][:, :, 1], dwt3_1_h[0][:, :, 2]), dim=1)

        return dwt1_1,dwt2_1,dwt3_1

class Dwt_MSEL1Loss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(Dwt_MSEL1Loss, self).__init__()
        #self.criterion1 = BMSAELoss()
        self.criterion2 = MSEL1Loss()
        self.dwt = Dwt()

    def __call__(self, input, target):
        input1,input2,input3=self.dwt(input)
        target1,target2,target3=self.dwt(target)
        return (self.criterion2(input1, target1)+self.criterion2(input2, target2)+self.criterion2(input3, target3)) / 3.

class G_loss(torch.nn.Module):
    def __init__(self):
       super(G_loss,self).__init__()
       self.criterion1 = BMSAELoss()
       self.criterion2 = torch.nn.BCELoss()

    def __call__(self, disc_g, pred, target):
       loss1 = self.criterion2(disc_g, torch.ones_like(disc_g))
       loss2 = self.criterion1(pred, target)
       loss_total = loss1+20*loss2
       return loss_total, loss1, loss2

class D_loss(torch.nn.Module):
    def __init__(self):
        super(D_loss, self).__init__()
        self.criterion2 = torch.nn.BCELoss()

    def __call__(self, disc_r, disc_g):
        loss_r = self.criterion2(disc_r, torch.ones_like(disc_r))
        loss_g = self.criterion2(disc_g, torch.zeros_like(disc_g))
        loss_total = loss_r + loss_g
        return loss_total
        
       







from torch.distributions import MultivariateNormal as MVN

def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    I = torch.eye(pred.shape[-1]).cuda()
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0)).cuda() # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    
    return loss

class BMCLoss(torch.nn.Module):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss_md(pred, target, noise_var)


