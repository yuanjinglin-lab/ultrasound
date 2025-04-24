import torch
import numpy as np
import os
import random
import pickle
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

class dog_and_cat(Dataset):

    def __init__(self, data_list, is_train=True):
        self.file_list = list(data_list)
        random.shuffle(self.file_list)
        self.phase = "train" if is_train else "validation"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        
        file_path = Path(self.file_list[index])
        class_name = file_path.name

        if  "cat" in class_name:
            label = 0
        elif "dog" in class_name:
            label = 1
        else:
            raise ValueError("pkl_name error")
        
        data = np.load(file_path)
        
        data = torch.tensor(data)
        
        label = torch.tensor(label)

        # if self.phase == "train":
        #     data = random_noise(data)
        #     data = random_crop_resize(data)
        #     data = random_flip(data)
            
        return data.float(), label.long()

class MyDataset(Dataset):

    def __init__(self, file_list, is_train=True):
        self.file_list = file_list
        self.phase = "train" if is_train else "validation"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        
        npy_path = self.file_list[index]
        pkl_name = os.path.basename(npy_path)

        if  "ABN" in pkl_name:
            label = 1
        elif "NOR" in pkl_name:
            label = 0
        else:
            raise ValueError("pkl_name error")
        label = torch.tensor(label)
        
        with open(npy_path, 'rb') as f:
            data = np.load(f)

        data = torch.tensor(data)
        # if self.phase == "train":
        #     data = random_noise(data)
        #     data = random_crop_resize(data)
        #     data = random_flip(data)
            
        N,T, C, H, W = data.shape
        data = data.reshape(N * T, C, H, W)
        return data.float(), label.long()

def random_flip(data):
    flip_flag = torch.rand(1)
    if flip_flag < 0.2:
        flip_dims = []
        for flip_i in range(2, 5):
            flip_i_flag = torch.rand(1)
            if flip_i_flag < 0.5:
                flip_dims.append(int(flip_i))
        if len(flip_dims) > 0:
            data = torch.flip(data, dims=flip_dims)

    return data

def random_crop_resize(data,rate=0.3):
    shape = data.shape[2:]
    crop_flag = torch.rand(1)
    if crop_flag < 0.3:
        x1 = np.random.randint(low=1, high=shape[0] * rate)
        x2 = np.random.randint(low=1, high=shape[0] * rate)
        
        y1 = np.random.randint(low=1, high=shape[1] * rate)
        z1 = np.random.randint(low=1, high=shape[1] * rate)
        y2 = np.random.randint(low=1, high=shape[2] * rate)
        z2 = np.random.randint(low=1, high=shape[2] * rate)
        data = data[:, :, x1:-x2, y1:-y2, z1:-z2]
        data = F.interpolate(data, shape)

    return data

def random_noise(data):
    noise_flag = torch.rand(1)
    if noise_flag < 0.3:
        max_mean = 0.01
        max_std = 0.05
        mean = torch.rand(1) * max_mean
        std = torch.rand(1) * max_std
        data = add_gaussian_noise(data, mean.item(), std.item())
    
    return data


def add_gaussian_noise(input, mean=0, std=1):
    # 生成与输入形状相同的高斯随机数张量
    noise = torch.normal(mean=mean, std=std, size=input.size()).to(input.device)
    # 将噪声加到输入上
    return input + noise