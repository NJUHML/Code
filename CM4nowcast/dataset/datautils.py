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

def normalize_nonan(x, scale, mean, reverse=False):
    """
    Normalize data or reverse normalization
    :param x: data array
    :param scale: const scaling value
    :param offset: const offset value
    :param reverse: boolean undo normalization
    :return: normalized x array
    """
    if reverse:
        # need to consider
        x[x==255.]=0
        return x*scale + mean
    else:
        x[x==255.]=0
        return (x-mean) / scale

def normalize(x, scale, mean, reverse=False):
    if reverse:
        return x*scale + mean
    else:
        return (x-mean) / scale

def datapreprocess1(x,y,scale=1,mean=0):
    x = seg_batch(x)
    y = seg_batch(y)
    x = normalize(x,scale,mean)
    y = normalize(y,scale,mean)
    return x,y

def datapreprocess(x,y,scale=1,mean=0, H = 128, W = 128):
    #assert len(x.shape) == 4
    #assert len(y.shape) == 4
    x = x.permute(0,3,1,2)
    y = y.permute(0,3,1,2)
    #x = torch.nn.functional.interpolate(x,size=(H,W),mode='bilinear')
    #y = torch.nn.functional.interpolate(y,size=(H,W),mode='bilinear')
    x = normalize(x,scale,mean)
    y = normalize(y,scale,mean)
    return x,y

def datapreprocess_npy(x,y,scale=1,mean=0, H = 128, W = 128):
    #assert len(x.shape) == 4
    #assert len(y.shape) == 4
    #x = x.permute(0,3,1,2)
    #y = y.permute(0,3,1,2)
    #x = torch.nn.functional.interpolate(x,size=(H,W),mode='bilinear')
    #y = torch.nn.functional.interpolate(y,size=(H,W),mode='bilinear')
    x = normalize(x,scale,mean)
    y = normalize(y,scale,mean)
    return x,y


def batch_optical_flow(inputs):
    uv = np.zeros([inputs.shape[0],2,inputs.shape[2],inputs.shape[3]])
    for i in range(inputs.shape[0]):
        uv[i]=pysteps.motion.lucaskanade.dense_lucaskanade(inputs[i])
    return torch.from_numpy(uv.astype(np.float32))
        

def datapreprocess_rover(x,y,scale=1,mean=0, H = 128, W = 128):
    #assert len(x.shape) == 4
    #assert len(y.shape) == 4
    x = x.permute(0,3,1,2)
    y = y.permute(0,3,1,2)
    x = torch.nn.functional.interpolate(x,size=(H,W),mode='bilinear')
    y = torch.nn.functional.interpolate(y,size=(H,W),mode='bilinear')
    uv = batch_optical_flow(x[:,-2:].numpy())
    #uv1 = batch_optical_flow(torch.cat([x[:,-1:],y[:,0:1]],axis=1).numpy())
    x = normalize(x[:,-1:],scale,mean)
    y = normalize(y[:,0:1],scale,mean)
    

    return torch.cat([x,uv],axis=1),y

def central_crop_sevir(x, length=128):
    return x[:,:,(192-(length//2)):(192-(length//2)+length),(192-(length//2)):(192-(length//2)+length)]
