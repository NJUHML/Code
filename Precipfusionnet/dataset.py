import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import netCDF4 as nc
import openpyxl

def random_sample_aligned(inputs, targets,sample_shape):
    # 随机生成采样位置
    while True:
        max_row_idx = inputs.shape[1] - sample_shape[1]
        max_col_idx = inputs.shape[2] - sample_shape[2]
        row_idx = np.random.randint(0, max_row_idx + 1)
        col_idx = np.random.randint(0, max_col_idx + 1)

        # 从两个数组中取出对应位置的子数组
        sample1 = inputs[:,row_idx:row_idx+sample_shape[1], col_idx:col_idx+sample_shape[2]]
        sample2 = targets[:,row_idx:row_idx+sample_shape[1], col_idx:col_idx+sample_shape[2]]
        count50 = np.sum(sample2 > 50)

        if count50 > sample_shape[1] * sample_shape[2] *0.01:
            return sample1, sample2,row_idx,col_idx

        
class MyDataSets_GFS2GPM_f024(Dataset):
    def __init__(self, data_type = 'train', input_type = 'P'):
        super(MyDataSets_GFS2GPM_f024, self).__init__()
        self.data_type = data_type
        self.input_type = input_type

    def __getitem__(self, idx):
        if self.data_type == "train":
            j=int(idx/16)
            inputs1 = np.load('/scratch/zhangzy/data/China_npy/GFS_f024/'+str(j)+".npy")
            targets = np.load('/scratch/zhangzy/data/China_npy/IMERG/'+str(j+4*0)+".npy")
            inputs1, targets = random_sample_aligned(inputs1, targets, (1,160, 256))
            a=0.0001
            inputs1 = np.where(inputs1 > 0, np.log(inputs1 + a) - np.log(a), 0)
            targets = np.where(targets > 0, np.log(targets + a) - np.log(a), 0)
            inputs1 = torch.from_numpy(inputs1.astype(np.float32))
            targets = torch.from_numpy(targets.astype(np.float32))
            inputs1 = inputs1/16.
            targets = targets/16.


        elif self.data_type == "val":
            j=int(idx)
            inputs1 = np.load('/scratch/zhangzy/data/China_npy/GFS_f024/'+str(j+4921+615)+".npy")
            targets = np.load('/scratch/zhangzy/data/China_npy/IMERG/'+str(j+4921+615+4*0)+".npy")

            a=0.0001
            inputs1,targets = inputs1[:,300:460,1000:1256],targets[:,300:460,1000:1256]
            inputs1 = np.where(inputs1 > 0, np.log(inputs1 + a) - np.log(a), 0)
            targets = np.where(targets > 0, np.log(targets + a) - np.log(a), 0)
            inputs1 = torch.from_numpy(inputs1.astype(np.float32))
            targets = torch.from_numpy(targets.astype(np.float32))
            inputs1 = inputs1/16.
            targets = targets/16.


        if self.input_type == 'P':
            return inputs1,targets


    def __len__(self):
        if self.data_type == "train":
            return int((4921+615)*16) #f024-4927
        if self.data_type == "val":
            return int(615) #f024-618


class MyDataSets_GFS2GPM_f024_norm(Dataset):
    def __init__(self, data_type = 'train', input_type = 'P'):
        super(MyDataSets_GFS2GPM_f024_norm, self).__init__()
        self.data_type = data_type
        self.input_type = input_type

    def __getitem__(self, idx):
        if self.data_type == "train":
            j=int(idx/16)
            inputs1 = np.load('/scratch/zhangzy/data/China_npy/GFS_f024/'+str(j)+".npy")
            targets = np.load('/scratch/zhangzy/data/China_npy/IMERG/'+str(j)+".npy")
            inputs1, targets = random_sample_aligned(inputs1, targets, (1,160, 256))
            inputs1 = torch.from_numpy(inputs1.astype(np.float32))
            targets = torch.from_numpy(targets.astype(np.float32))
            inputs1 = inputs1/1000.
            targets = targets/1000.


        elif self.data_type == "val":
            j=int(idx)
            inputs1 = np.load('/scratch/zhangzy/data/China_npy/GFS_f024/'+str(j+4921)+".npy")
            targets = np.load('/scratch/zhangzy/data/China_npy/IMERG/'+str(j+4921)+".npy")
            inputs1,targets = inputs1[:,300:460,1000:1256],targets[:,300:460,1000:1256]
            inputs1 = torch.from_numpy(inputs1.astype(np.float32))
            targets = torch.from_numpy(targets.astype(np.float32))
            inputs1 = inputs1/1000.
            targets = targets/1000.

        if self.input_type == 'P':
            return inputs1,targets

    def __len__(self):
        if self.data_type == "train":
            return int((4921+615)*16) #f024-4927
        if self.data_type == "val":
            return int(615) #f024-618
        
        

def load_dataset(data_type = None, input_type = None, time_type = None):
    

    if data_type == 'GFS2GPM_f024':
        train_data = MyDataSets_GFS2GPM_f024(data_type = 'train', input_type=input_type)
        val_data = MyDataSets_GFS2GPM_f024(data_type = 'val', input_type=input_type)

    elif data_type == 'GFS2GPM_f024_norm':
        train_data = MyDataSets_GFS2GPM_f024_norm(data_type = 'train', input_type=input_type)
        val_data = MyDataSets_GFS2GPM_f024_norm(data_type = 'val', input_type=input_type)


    return train_data, val_data

