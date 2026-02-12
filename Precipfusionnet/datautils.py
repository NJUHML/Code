import numpy as np
import pysteps
import torch

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

def seg_batch(images):
    bs = images.size(0)
    frames = images.size(1)
    hight = images.size(2)
    weight = images.size(3)
    times = images.size(4)
    return  images.reshape(bs*frames,hight,weight,times)


def normalize(x, scale, mean, reverse=False):
    if reverse:
        return x*scale + mean
    else:
        return (x-mean) / scale

def batch_optical_flow(inputs):
    uv = np.zeros([inputs.shape[0],2,inputs.shape[2],inputs.shape[3]])
    for i in range(inputs.shape[0]):
        uv[i]=pysteps.motion.lucaskanade.dense_lucaskanade(inputs[i])
    return torch.from_numpy(uv.astype(np.float32))
        