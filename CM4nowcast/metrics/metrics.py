import pysteps
import torch 
import numpy as np
from skimage.metrics import structural_similarity


def CalSSIM(image1, image2):
    K1 = 0.01
    K2 = 0.03
    L = 255


    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    sigma1 = np.std(image1)
    sigma2 = np.std(image2)
    sigma12 = np.cov(image1.flatten(), image2.flatten())[0, 1]


    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2)
    ssim_value = numerator / denominator

    return ssim_value

def torch_numpy(x):
    return x.numpy()

def torchcuda_numpy(x):
    return x.detach().cpu().numpy()



def PSD(inputs):
    batch = inputs.shape[0]
    time_step = inputs.shape[1]
    PSD = []
    for i in range(batch):
        PSD1 = []
        for j in range(time_step):
             _,fre1 = pysteps.utils.spectral.rapsd(inputs[i,j], fft_method=np.fft, return_freq=True)
             PSD1.append(fre1)
        PSD.append(PSD1)
    return np.array(PSD)


def RMSECascade(pred,target,cascade):
    filter1 = pysteps.cascade.bandpass_filters.filter_gaussian([384,384], cascade)
    func = pysteps.verification.interface.get_method('RMSE', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    rmse = []
    for i in range(batch):
        rmse1 = []
        for j in range(time_step):
            rmse2 = []
            decompred = pysteps.cascade.decomposition.decomposition_fft(pred[i,j,:,:],filter1)
            decomtarget = pysteps.cascade.decomposition.decomposition_fft(target[i,j,:,:],filter1)
            for k in range(cascade):
                rmse2.append(func(decompred['cascade_levels'][k],decomtarget['cascade_levels'][k])['RMSE'])
            rmse1.append(rmse2)
        rmse.append(rmse1)
    return np.array(rmse)




def CSI(pred,target,threshold):
    func = pysteps.verification.interface.get_method('CSI', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    csi = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            csi[i,j]=func(pred[i,j],target[i,j],thr=threshold)['CSI']
    return csi

def ETS(pred,target,threshold):
    func = pysteps.verification.interface.get_method('GSS', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    ets = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            ets[i,j]=func(pred[i,j],target[i,j],thr=threshold)['GSS']     
    return ets

def POD(pred,target,threshold):
    func = pysteps.verification.interface.get_method('POD', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    pod = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            pod[i,j]=func(pred[i,j],target[i,j],thr=threshold)['POD']
    return pod

def FAR(pred,target,threshold):
    func = pysteps.verification.interface.get_method('FAR', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    far = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            far[i,j]=func(pred[i,j],target[i,j],thr=threshold)['FAR']
    return far

def HSS(pred,target,threshold):
    func = pysteps.verification.interface.get_method('HSS', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    hss = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            hss[i,j]=func(pred[i,j],target[i,j],thr=threshold)['HSS']
    return hss

def BIAS(pred,target,threshold):
    func = pysteps.verification.interface.get_method('BIAS', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    bias = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            bias[i,j]=func(pred[i,j],target[i,j],thr=threshold)['BIAS']
    return bias


def FSS(pred,target,threshold,scale=1):
    func = pysteps.verification.interface.get_method('FSS', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    fss = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            fss[i,j]=func(pred[i,j],target[i,j],thr=threshold,scale=scale)
    return fss


def NMSE(pred,target):
    func = pysteps.verification.interface.get_method('NMSE', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    nmse = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            nmse[i,j]=func(pred[i,j],target[i,j])['NMSE']
    return nmse

def MSE(pred,target):
    func = pysteps.verification.interface.get_method('MSE', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    mse = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            mse[i,j]=func(pred[i,j],target[i,j])['MSE']
    return mse


def MAE(pred,target):
    func = pysteps.verification.interface.get_method('MAE', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    mae = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            mae[i,j]=func(pred[i,j],target[i,j])['MAE']
    return mae

def corr_p(pred,target):
    func = pysteps.verification.interface.get_method('corr_p', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    corr = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            corr[i,j]=func(pred[i,j],target[i,j])['corr_p']
    return corr

def corr_s(pred,target):
    func = pysteps.verification.interface.get_method('corr_s', type='deterministic')
    batch = pred.shape[0]
    time_step = pred.shape[1]
    corr = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            corr[i,j]=func(pred[i,j],target[i,j])['corr_s']
    return corr


def corr_e_basic(pred,target):
    a = np.sum(np.multiply(pred,target))
    b = (np.sum(pred**2)*np.sum(target**2))**(0.5)
    if b == 0:
        return 0
    else:
        return a/b

def corr_e(pred,target):
    batch = pred.shape[0]
    time_step = pred.shape[1] 
    corr = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            corr[i,j]=corr_e_basic(pred[i,j],target[i,j])
    return corr 


def SSIM_func(pred,target):
    batch = pred.shape[0]
    time_step = pred.shape[1]
    ssim = np.zeros([batch,time_step])
    for i in range(batch):
        for j in range(time_step):
            ssim[i,j]=structural_similarity(pred[i,j],target[i,j], data_range=1.0)
    return ssim

