import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch
import random



def reshape_patch(images, patch_size):
    bs = images.size(0)
    nc = images.size(1)
    height = images.size(2)
    weight = images.size(3)
    x = images.reshape(bs, nc, int(height / patch_size), patch_size, int(weight / patch_size), patch_size)
    x = x.transpose(2, 5)
    x = x.transpose(4, 5)
    x = x.reshape(bs, nc * patch_size * patch_size, int(height / patch_size), int(weight / patch_size))

    return x


def reshape_patch_back(images, patch_size):
    bs = images.size(0)
    nc = int(images.size(1) / (patch_size * patch_size))
    height = images.size(2)
    weight = images.size(3)
    x = images.reshape(bs, nc, patch_size, patch_size, height, weight)
    x = x.transpose(4, 5)
    x = x.transpose(2, 5)
    x = x.reshape(bs, nc, height * patch_size, weight * patch_size)

    return x


def load_dataset(opt):
    if opt.dataset == 'predmnist':
        import sys
        sys.path.append('/root/code')
        from sevir.dataset.dataset import MyTrainDataSets_sevir
        train_data, valid_data = 0, 0
        test_data = MyTrainDataSets_sevir("/root/autodl-tmp/SEVIR/test_384/", seq_len = 13, pre_len = 12)

    return train_data, valid_data, test_data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


import torch
import torch.fft as fft

def lowpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1).cuda()
    
    fft_input = fft.rfft2(input)
    return fft.irfft2(fft_input * kernel, s=input.shape[-2:])

def highpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) > limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) > limit
    kernel = torch.outer(pass2, pass1).cuda()

    fft_input = fft.rfft2(input)
    return fft.irfft2(fft_input * kernel, s=input.shape[-2:])


def lowpass_torch_6(inputs):
    inputs[:,0]= lowpass_torch(inputs[:,0],1/8)
    inputs[:,1]= lowpass_torch(inputs[:,1],1/4)
    inputs[:,2]= lowpass_torch(inputs[:,2],1/2)
    #inputs[:,3]= lowpass_torch(inputs[:,3],1/2)
    return inputs

def lowpass_torch_12(inputs):
    #inputs[:,0]= lowpass_torch(inputs[:,0],1/16)
    inputs[:,0]= lowpass_torch(inputs[:,0],1/8)
    inputs[:,1]= lowpass_torch(inputs[:,1],1/4)
    inputs[:,2]= lowpass_torch(inputs[:,2],1/2)
    return inputs

def get_scheduler(optimizer, opt, t_max):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)

    elif opt.lr_policy == 'warmup':
        warmup_iter = opt.epoch_size*0.2
        warmup_scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=warmup_lambda(warmup_steps=warmup_iter,min_lr_ratio=0.0))
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.epoch_size - warmup_iter), eta_min=1e-3 * opt.lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter])

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def get_scheduler_no_opt(optimizer, opt, t_max):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)

    elif opt.lr_policy == 'warmup':
        warmup_iter = opt.epoch_size*0.2
        warmup_scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=warmup_lambda(warmup_steps=warmup_iter,min_lr_ratio=0.0))
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.epoch_size - warmup_iter), eta_min=1e-3 * opt.lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter])

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)



