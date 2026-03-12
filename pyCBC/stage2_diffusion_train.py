import os
import random
import numpy as np
import math
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from einops import rearrange
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
from netCDF4 import Dataset as ncDataset
import glob
from functools import partial
import argparse
from collections import defaultdict
import re

Tensor = torch.Tensor


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Unified configuration class managing all hyperparameters and paths."""

    # ==================== Data Paths ====================
    class DataPaths:
        DATA_FOLDER = "/data/guxh/R/"
        SAVE_DIR = '/data/guxh/test1002/'

        LATEST_CHECKPOINT_NAME = "latest_checkpoint_standard.pth"
        BEST_MODEL_NAME = "best_model_gpm_standard.pth"

    # ==================== Data Processing ====================
    class DataProcessing:
        # Spatial domain
        LON_RANGE = (110, 130)
        LAT_RANGE = (20, 40)

        # Precipitation normalization
        EPSILON = 1e-6         # small constant to avoid log(0)
        X_MAX = 121            # maximum precipitation value (mm), used for normalization

        # Sample filtering
        MIN_PRECIP_THRESHOLD = 0.1   # minimum domain-mean precipitation to retain a sample (mm)
        FILTER_SAMPLES = True

        # 6-hour accumulation window
        USE_6HOUR = True

    # ==================== Model Architecture ====================
    class Model:
        DIM_IN = 1
        DIM_OUT = 1
        BASE_CHANNELS = 128           # base UNet channel width
        CHANNEL_MULTS = (1, 2, 4, 8)
        RESNET_BLOCK_GROUPS = 4
        SCALE = [2, 2, 2]             # down/up-sample factors per stage
        LEARN_VARIANCE = False        # True = Improved DDPM (learned variance)

    # ==================== Diffusion Process ====================
    class Diffusion:
        TIMESTEPS = 1000
        SCHEDULE = 'cosine'           # 'cosine' or 'linear'
        # Linear schedule params (used only when SCHEDULE='linear')
        LINEAR_BETA_START = 0.0001
        LINEAR_BETA_END = 0.02
        # Cosine schedule param
        COSINE_S = 0.008

    # ==================== Training Hyperparameters ====================
    class Training:
        N_EPOCHS = 200
        BATCH_SIZE = 5
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 1e-6
        GRAD_CLIP_NORM = 1.0
        NUM_WORKERS = 2
        PIN_MEMORY = True
        DEVICE = 'cuda:4'

        # Checkpoint save frequency (every N epochs)
        SAVE_EVERY_N_EPOCHS = 10
        # Batch-level checkpoint save frequency
        SAVE_EVERY_N_BATCHES = 100
        # VRAM flush frequency (every N batches)
        FLUSH_EVERY_N_BATCHES = 50

    # ==================== LR Scheduler ====================
    class Scheduler:
        MODE = 'min'
        FACTOR = 0.8
        PATIENCE = 5

    # ==================== Visualization ====================
    class Visualization:
        # Noise timesteps at which to visualize denoising quality during training
        TEST_NOISE_STEPS = [100, 300, 500, 700, 900, 999]
        # Number of random samples to visualize per epoch
        NUM_VIS_SAMPLES = 3
        # Precipitation colormap levels for visualization (mm)
        PRECIP_LEVELS = [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0]
        VIS_DPI = 150


# ============================================================================
# Shorthand aliases (keep code below readable)
# ============================================================================
epsilon = Config.DataProcessing.EPSILON
x_max   = Config.DataProcessing.X_MAX


# ===================== Data Processing =====================
def custom_normalize(precip):
    """Log-based normalization mapping precipitation values to [-1, 1]."""
    precip = np.where(precip < 0, 0, precip)
    non_zero_mask = (precip > 0)
    precip_non_zero = precip[non_zero_mask]
    if len(precip_non_zero) > 0:
        precip_non_zero = (2 * np.log(1 + precip_non_zero / epsilon)) / np.log(1 + x_max / epsilon) - 1
        precip[non_zero_mask] = precip_non_zero
    return precip

def custom_denormalize(tensor):
    """Inverts custom_normalize to recover precipitation values in mm."""
    if torch.is_tensor(tensor):
        tensor = tensor.squeeze().cpu().numpy()
    tensor = (tensor + 1) * np.log(1 + x_max / epsilon) / 2
    tensor = np.exp(tensor) - 1
    tensor = tensor * epsilon
    tensor = np.where(tensor < 0, 0, tensor)
    return tensor

def parse_filename_timestamp(filename):
    """Parses date and time components from a GPM IMERG HDF5/nc4 filename."""
    pattern = r'3B-HHR\.MS\.MRG\.3IMERG\.(\d{8})-S(\d{6})-E(\d{6})\.\d{4}\.V\d{2}B\.HDF5\.SUB\.nc4'
    match = re.match(pattern, os.path.basename(filename))
    if match:
        date_str = match.group(1)
        start_time = match.group(2)
        end_time = match.group(3)
        return date_str, start_time, end_time
    else:
        raise ValueError(f"Cannot parse filename: {filename}")

def group_files_by_hour(file_paths):
    """Groups GPM IMERG 30-minute files by hour; returns only hours with exactly 2 files."""
    hour_groups = defaultdict(list)
    for file_path in file_paths:
        try:
            date_str, start_time, end_time = parse_filename_timestamp(file_path)
            start_hour = start_time[:2]
            hour_key = f"{date_str}_{start_hour}"
            hour_groups[hour_key].append({
                'path': file_path,
                'date': date_str,
                'start_time': start_time,
                'end_time': end_time,
                'hour': start_hour
            })
        except ValueError as e:
            print(f"Skipping file {file_path}: {e}")
            continue
    
    complete_hours = {}
    for hour_key, files in hour_groups.items():
        if len(files) == 2:
            files.sort(key=lambda x: x['start_time'])
            complete_hours[hour_key] = files
        else:
            print(f"Warning: hour {hour_key} has {len(files)} file(s), skipping")
    
    print(f"Found {len(complete_hours)} complete hourly records")
    
    sorted_hour_keys = sorted(complete_hours.keys())
    
    return complete_hours, sorted_hour_keys

def load_and_combine_hourly_data(hour_files):
    """Loads two 30-minute GPM files for one hour and returns their mean precipitation field."""
    precip_data = []
    lon_min, lon_max = Config.DataProcessing.LON_RANGE
    lat_min, lat_max = Config.DataProcessing.LAT_RANGE
    for file_info in hour_files:
        file_path = file_info['path']
        with ncDataset(file_path, 'r') as ds:
            precip = ds.variables['precipitation'][:].astype(np.float32)
            lat = ds.variables['lat'][:].astype(np.float32)
            lon = ds.variables['lon'][:].astype(np.float32)
            
            lat_mask = (lat >= lat_min) & (lat <= lat_max)
            precip = precip[:, :, lat_mask]
            lon_mask = (lon >= lon_min) & (lon <= lon_max)
            precip = precip[:,lon_mask, :] 
            
            if len(precip.shape) == 3:
                precip = precip[0]
            elif len(precip.shape) == 2:
                pass
            else:
                raise ValueError(f"Unexpected data shape: {precip.shape}")
            precip_data.append(precip)
    
    hourly_precip = np.mean(precip_data, axis=0)
    return hourly_precip

class GPMDataHourly(Dataset):
    def __init__(self, folder_path, file_paths=None, transform=None, verbose=False, 
                 min_precip_threshold=None, filter_samples=None, use_6hour=None,
                 skip_init=False):
        """
        GPM IMERG precipitation dataset.

        Args:
            folder_path: path to data directory
            file_paths: optional explicit list of file paths
            transform: optional data transform
            verbose: enable verbose output
            min_precip_threshold: minimum mean precipitation to retain a sample (mm)
            filter_samples: whether to discard low-precipitation samples
            use_6hour: use 6-hour accumulation windows instead of 1-hour
            skip_init: skip data processing on init (used to avoid reprocessing in training loop)
        """
        self.folder_path = folder_path
        self.transform = transform
        self.verbose = verbose
        self.min_precip_threshold = min_precip_threshold if min_precip_threshold is not None \
            else Config.DataProcessing.MIN_PRECIP_THRESHOLD
        self.filter_samples = filter_samples if filter_samples is not None \
            else Config.DataProcessing.FILTER_SAMPLES
        self.use_6hour = use_6hour if use_6hour is not None \
            else Config.DataProcessing.USE_6HOUR
        
        if file_paths is not None:
            all_file_paths = file_paths
        else:
            all_file_paths = sorted(glob.glob(f"{folder_path}/*.nc4"))
        
        self.hour_groups, sorted_hour_keys = group_files_by_hour(all_file_paths)
        
        if skip_init:
            self.six_hour_samples = []
            self.hour_keys = []
            return
        
        if self.use_6hour:
            if self.filter_samples:
                self.six_hour_samples = create_6hour_samples(
                    self.hour_groups, 
                    sorted_hour_keys, 
                    min_precip_threshold=self.min_precip_threshold
                )
            else:
                self.six_hour_samples = create_6hour_samples(
                    self.hour_groups, 
                    sorted_hour_keys, 
                    min_precip_threshold=0.0
                )
            print(f"Training on {len(self.six_hour_samples)} 6-hour precipitation samples")
        else:
            self.hour_keys = sorted_hour_keys
            print(f"Found {len(self.hour_keys)} hourly records")
            
            if self.filter_samples:
                self._filter_low_precipitation_samples()
            
            print(f"Training on {len(self.hour_keys)} 1-hour precipitation samples")

    def _filter_low_precipitation_samples(self):
        """Removes 1-hour samples whose mean precipitation falls below the threshold."""
        print(f"Filtering samples with mean precipitation < {self.min_precip_threshold} mm ...")
        filtered_hour_keys = []
        filtered_count = 0
        
        for hour_key in tqdm(self.hour_keys, desc="Filtering samples"):
            hour_files = self.hour_groups[hour_key]
            try:
                precip = load_and_combine_hourly_data(hour_files)
                mean_precip = np.mean(precip)
                if mean_precip >= self.min_precip_threshold:
                    filtered_hour_keys.append(hour_key)
                else:
                    filtered_count += 1
            except Exception as e:
                print(f"Error processing {hour_key}: {e}, skipping")
                filtered_count += 1
                continue
        
        self.hour_keys = filtered_hour_keys
        print(f"Filtering complete: removed {filtered_count}, kept {len(self.hour_keys)} samples")

    def __len__(self):
        if self.use_6hour:
            return len(self.six_hour_samples)
        else:
            return len(self.hour_keys)

    def __getitem__(self, idx):
        if self.use_6hour:
            sample = self.six_hour_samples[idx]
            precip = sample['precip_data']
        else:
            hour_key = self.hour_keys[idx]
            hour_files = self.hour_groups[hour_key]
            precip = load_and_combine_hourly_data(hour_files)
        
        precip = custom_normalize(precip)
        precip = precip[np.newaxis, :, :]
        precip = torch.tensor(precip, dtype=torch.float32)
        return precip
    
def create_6hour_samples(hour_groups, sorted_hour_keys, min_precip_threshold=0.1):
    """
    Builds 6-hour accumulated precipitation samples using a sliding window.
    Only consecutive hourly blocks whose individual mean precipitation exceeds
    min_precip_threshold are included.
    Returns a list of valid 6-hour sample dicts.
    """
    print(f"Creating 6-hour samples (sliding window, threshold={min_precip_threshold} mm)...")
    
    valid_hour_keys = []
    valid_hour_data = {}
    
    print("Step 1: filter 1-hour samples...")
    for hour_key in tqdm(sorted_hour_keys, desc="Filtering 1-hour samples"):
        hour_files = hour_groups[hour_key]
        try:
            precip = load_and_combine_hourly_data(hour_files)
            mean_precip = np.mean(precip)
            if mean_precip >= min_precip_threshold:
                valid_hour_keys.append(hour_key)
                valid_hour_data[hour_key] = precip
        except Exception as e:
            print(f"Error processing {hour_key}: {e}, skipping")
            continue
    
    print(f"{len(valid_hour_keys)} valid 1-hour samples after filtering")
    
    six_hour_samples = []
    
    print("Step 2: build 6-hour samples...")
    for i in tqdm(range(len(valid_hour_keys) - 5), desc="Building 6-hour samples"):
        current_6hours = valid_hour_keys[i:i+6]
        
        is_continuous = True
        for j in range(5):
            current_key = current_6hours[j]
            next_key = current_6hours[j+1]
            
            current_date = current_key.split('_')[0]
            current_hour = int(current_key.split('_')[1])
            next_date = next_key.split('_')[0]
            next_hour = int(next_key.split('_')[1])
            
            if current_date == next_date:
                if next_hour != current_hour + 1:
                    is_continuous = False
                    break
            else:
                from datetime import datetime, timedelta
                current_datetime = datetime.strptime(f"{current_date}{current_hour:02d}", "%Y%m%d%H")
                next_datetime = datetime.strptime(f"{next_date}{next_hour:02d}", "%Y%m%d%H")
                if (next_datetime - current_datetime).total_seconds() != 3600:
                    is_continuous = False
                    break
        
        if is_continuous:
            try:
                precip_sum = np.zeros_like(valid_hour_data[current_6hours[0]])
                for hour_key in current_6hours:
                    precip_sum += valid_hour_data[hour_key]
                
                six_hour_samples.append({
                    'hour_keys': current_6hours,
                    'precip_data': precip_sum,
                    'start_time': current_6hours[0],
                    'end_time': current_6hours[-1]
                })
            except Exception as e:
                print(f"Failed to build 6-hour sample {current_6hours}: {e}")
                continue
    
    print(f"Successfully built {len(six_hour_samples)} 6-hour samples")
    return six_hour_samples


# ===================== UNet Model =====================
def Upsample(dim_in, dim_out, scale=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
        nn.Conv2d(dim_in, dim_out, 3, padding=1),
    )

def Downsample(dim_in, dim_out, scale=2):
    return nn.Sequential(
        nn.MaxPool2d(scale),
        nn.Conv2d(dim_in, dim_out, 1),
    )

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] * 1000.
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out, eps=1e-5)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp_t = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) 
                     if time_emb_dim is not None else None)
        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift_t = None
        if self.mlp_t is not None and time_emb is not None:
            time_emb = self.mlp_t(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift_t = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift_t)
        h = self.block2(h)
        return h + self.res_conv(x)

class UNet(nn.Module):
    def __init__(self, dim_in=None, dim_out=None, c=None, c_mults=None, 
                 resnet_block_groups=None, scale=None, learn_variance=None):
        super().__init__()

        if dim_in is None:
            dim_in = Config.Model.DIM_IN
        if dim_out is None:
            dim_out = Config.Model.DIM_OUT
        if c is None:
            c = Config.Model.BASE_CHANNELS
        if c_mults is None:
            c_mults = Config.Model.CHANNEL_MULTS
        if resnet_block_groups is None:
            resnet_block_groups = Config.Model.RESNET_BLOCK_GROUPS
        if scale is None:
            scale = Config.Model.SCALE
        if learn_variance is None:
            learn_variance = Config.Model.LEARN_VARIANCE

        self.learn_variance = learn_variance
        self.out_channels = dim_out * 2 if learn_variance else dim_out
        
        self.init_conv = nn.Conv2d(dim_in, c, 1, padding=0)
        dims = [c*x for x in c_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
        time_dim = c * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    block_klass(d_in, d_in, time_emb_dim=time_dim),
                    block_klass(d_in, d_in, time_emb_dim=time_dim),
                    Downsample(d_in, d_out, scale=scale[ind])
                    if not is_last
                    else nn.Conv2d(d_in, d_out, 3, padding=1),
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    block_klass(d_out + d_in, d_out, time_emb_dim=time_dim),
                    block_klass(d_out + d_in, d_out, time_emb_dim=time_dim),
                    Upsample(d_out, d_in, scale=scale[-ind-1])
                    if not is_last
                    else nn.Conv2d(d_out, d_in, 3, padding=1),
                ])
            )

        self.final_res_block = block_klass(c * 2, c, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(c, self.out_channels, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        return x


# ===================== DDPM Implementation =====================
def cosine_beta_schedule(timesteps, s=None):
    """
    Cosine noise schedule from the Improved DDPM paper (Nichol & Dhariwal, 2021).
    Provides smoother noise decay and avoids signal collapse at high timesteps.
    """
    if s is None:
        s = Config.Diffusion.COSINE_S
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=None, beta_end=None):
    """Linear noise schedule from the original DDPM paper (Ho et al., 2020)."""
    if beta_start is None:
        beta_start = Config.Diffusion.LINEAR_BETA_START
    if beta_end is None:
        beta_end = Config.Diffusion.LINEAR_BETA_END
    return torch.linspace(beta_start, beta_end, timesteps)

class DDPM:
    """
    Standard and Improved DDPM implementation.
    Based on:
      Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
      Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)
    """
    def __init__(self, timesteps=None, device=None, 
                 learn_variance=None, schedule=None):

        if timesteps is None:
            timesteps = Config.Diffusion.TIMESTEPS
        if device is None:
            device = Config.Training.DEVICE
        if learn_variance is None:
            learn_variance = Config.Model.LEARN_VARIANCE
        if schedule is None:
            schedule = Config.Diffusion.SCHEDULE

        self.device = device
        self.timesteps = timesteps
        self.learn_variance = learn_variance
        
        print(f"Config: learn_variance={learn_variance}, schedule={schedule}")
        
        if schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps).to(device)
        elif schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        print(f"Beta range: [{self.betas.min().item():.6f}, {self.betas.max().item():.6f}]")
        print(f"Alpha_cumprod range: [{self.alphas_cumprod.min().item():.6f}, {self.alphas_cumprod.max().item():.6f}]")

    def extract(self, a, t, x_shape):
        """
        Gathers values from tensor a at indices t and reshapes to broadcast over x_shape.
        """
        batch_size = t.shape[0]
        device = t.device
        a = a.to(device)
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion q(x_t | x_0):
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """Recovers x_0 estimate from predicted noise."""
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def q_posterior(self, x_start, x_t, t):
        """
        Computes the mean and variance of the true posterior q(x_{t-1} | x_t, x_0).
        """
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, model, x_t, t):
        """
        Decodes model output into noise prediction, x_0 estimate, and variance.
        Returns: pred_noise, pred_x0, model_variance
        """
        model_output = model(x_t, t)
        
        if self.learn_variance:
            model_output, model_var_values = torch.split(model_output, x_t.shape[1], dim=1)
            
            min_log = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
            max_log = self.extract(torch.log(self.betas), t, x_t.shape)
            
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance = self.extract(self.posterior_variance, t, x_t.shape)
        
        pred_noise = model_output
        pred_x0 = self.predict_start_from_noise(x_t, t, pred_noise)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        return pred_noise, pred_x0, model_variance

    def p_mean_variance(self, model, x_t, t):
        """Computes the mean and variance of the reverse process p(x_{t-1} | x_t)."""
        pred_noise, pred_x0, model_variance = self.model_predictions(model, x_t, t)
        
        model_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0 +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        
        return model_mean, model_variance, pred_x0

    def p_losses(self, model, x_start, t, noise=None):
        """
        Computes training loss.
        Standard DDPM: simple MSE on predicted noise.
        Improved DDPM: MSE + weighted VLB loss for variance learning.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        pred_noise, pred_x0, model_variance = self.model_predictions(model, x_noisy, t)
        
        simple_loss = F.mse_loss(noise, pred_noise)
        
        if self.learn_variance:
            true_mean, true_variance, true_log_variance = self.q_posterior(
                x_start, x_noisy, t
            )
            
            model_mean, _, _ = self.p_mean_variance(model, x_noisy, t)
            model_log_variance = torch.log(torch.clamp(model_variance, min=1e-20))
            
            kl = 0.5 * (
                -1.0
                + model_log_variance
                - true_log_variance
                + true_variance / model_variance
                + ((true_mean - model_mean) ** 2) / model_variance
            )
            kl = kl.mean(dim=[1, 2, 3])
            
            decoder_nll = -self._log_normal(x_start, means=model_mean, log_scales=0.5 * model_log_variance)
            decoder_nll = decoder_nll.mean(dim=[1, 2, 3])
            
            vb_loss = torch.where(t == 0, decoder_nll, kl)
            
            loss = simple_loss + 0.001 * vb_loss.mean()
        else:
            loss = simple_loss
        
        return loss

    def _log_normal(self, x, means, log_scales):
        """Computes the log-probability of x under a Gaussian with given mean and log-scale."""
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        normalized_x = centered_x * inv_stdv
        log_probs = -0.5 * (normalized_x ** 2 + 2 * log_scales + np.log(2 * np.pi))
        return log_probs

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Single reverse step: samples x_{t-1} from x_t."""
        model_mean, model_variance, pred_x0 = self.p_mean_variance(model, x, t)
        
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Full reverse diffusion loop starting from Gaussian noise."""
        device = next(model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            img = self.p_sample(
                model, img, 
                torch.full((b,), i, device=device, dtype=torch.long), 
                i
            )
        
        return img

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1):
        """Generates new samples by running the full reverse diffusion loop."""
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


# ===================== Checkpoint Utilities =====================
def save_checkpoint(model, optimizer, scheduler, epoch, loss, train_losses, save_dir, 
                   is_best=False, learn_variance=False, schedule='cosine'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'train_losses': train_losses,
        'x_max': x_max,
        'learn_variance': learn_variance,
        'schedule': schedule,
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'pytorch_version': torch.__version__,
    }
    
    latest_path = os.path.join(save_dir, Config.DataPaths.LATEST_CHECKPOINT_NAME)
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved to {latest_path}")
    
    if is_best:
        best_path = os.path.join(save_dir, Config.DataPaths.BEST_MODEL_NAME)
        torch.save(checkpoint, best_path)
        print(f"Best model checkpoint saved to {best_path}")
    
    if (epoch + 1) % Config.Training.SAVE_EVERY_N_EPOCHS == 0:
        epoch_path = os.path.join(save_dir, f"checkpoint_standard_epoch_{epoch+1}.pth")
        torch.save(checkpoint, epoch_path)
        print(f"Epoch checkpoint saved to {epoch_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    train_losses = checkpoint.get('train_losses', [])
    learn_variance = checkpoint.get('learn_variance', False)
    schedule = checkpoint.get('schedule', 'cosine')
    
    try:
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
        if 'numpy_random_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_random_state'])
        if 'torch_random_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_random_state'])
    except Exception as e:
        print(f"Warning: could not fully restore RNG state: {e}")
    
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    print(f"Best loss: {best_loss:.6f}, learn_variance: {learn_variance}, schedule: {schedule}")
    
    return start_epoch, best_loss, train_losses, learn_variance, schedule

def find_latest_checkpoint(save_dir):
    latest_path = os.path.join(save_dir, Config.DataPaths.LATEST_CHECKPOINT_NAME)
    if os.path.exists(latest_path):
        return latest_path
    
    checkpoints = glob.glob(os.path.join(save_dir, "checkpoint_standard_epoch_*.pth"))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return checkpoints[-1]
    
    return None


# ===================== Main Training Function =====================
def train_gpm_standard(resume_from=None, learn_variance=None, schedule=None,
                       min_precip_threshold=None, filter_samples=None):
    """
    Trains a Standard DDPM or Improved DDPM on 6-hour GPM IMERG precipitation data.

    Args:
        resume_from: path to a checkpoint to resume from
        learn_variance: if True, enables Improved DDPM with learned variance
        schedule: noise schedule type ('cosine' or 'linear')
        min_precip_threshold: minimum mean precipitation for sample inclusion (mm)
        filter_samples: whether to discard low-precipitation samples
    """
    # Apply defaults from Config if not overridden by CLI
    if learn_variance is None:
        learn_variance = Config.Model.LEARN_VARIANCE
    if schedule is None:
        schedule = Config.Diffusion.SCHEDULE
    if min_precip_threshold is None:
        min_precip_threshold = Config.DataProcessing.MIN_PRECIP_THRESHOLD
    if filter_samples is None:
        filter_samples = Config.DataProcessing.FILTER_SAMPLES

    n_epoch      = Config.Training.N_EPOCHS
    batch_size   = Config.Training.BATCH_SIZE
    timesteps    = Config.Diffusion.TIMESTEPS
    lrate        = Config.Training.LEARNING_RATE
    save_dir     = Config.DataPaths.SAVE_DIR
    data_folder  = Config.DataPaths.DATA_FOLDER

    TEST_NOISE_STEPS = Config.Visualization.TEST_NOISE_STEPS
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(Config.Training.DEVICE if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    model_label = 'Improved DDPM' if learn_variance else 'Standard DDPM'
    print(f"Training {model_label} on 6-hour precipitation samples")
    print(f"learn_variance: {learn_variance}, schedule: {schedule}")
    print("=" * 70)
    
    # Instantiate model
    model = UNet(learn_variance=learn_variance).to(device)
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lrate, weight_decay=Config.Training.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.Scheduler.MODE,
        factor=Config.Scheduler.FACTOR,
        patience=Config.Scheduler.PATIENCE
    )
    
    # DDPM process
    ddpm = DDPM(
        timesteps=timesteps, 
        device=device,
        learn_variance=learn_variance,
        schedule=schedule
    )
    
    # Dataset
    all_file_paths = sorted(glob.glob(f"{data_folder}/*.nc4"))
    
    print("\nInitializing dataset and building 6-hour samples...")
    persistent_dataset = GPMDataHourly(
        folder_path=data_folder, 
        file_paths=all_file_paths,
        min_precip_threshold=min_precip_threshold,
        filter_samples=filter_samples,
        use_6hour=True
    )
    
    filtered_hour_groups = persistent_dataset.hour_groups
    if persistent_dataset.use_6hour:
        filtered_six_hour_samples = persistent_dataset.six_hour_samples
    else:
        filtered_hour_keys = persistent_dataset.hour_keys
    
    # Initialize training state
    start_epoch = 0
    best_loss = float('inf')
    train_losses = []
    
    # Resume from checkpoint if available
    checkpoint_path = resume_from if resume_from else find_latest_checkpoint(save_dir)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            result = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
            start_epoch, best_loss, train_losses, loaded_lv, loaded_schedule = result
            
            if loaded_lv != learn_variance or loaded_schedule != schedule:
                print(f"Warning: checkpoint config does not match current config!")
                print(f"  Checkpoint: learn_variance={loaded_lv}, schedule={loaded_schedule}")
                print(f"  Current:    learn_variance={learn_variance}, schedule={schedule}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            start_epoch = 0
            best_loss = float('inf')
            train_losses = []
    
    print(f"Starting from epoch {start_epoch+1}/{n_epoch}")
    print(f"Current best loss: {best_loss:.6f}")
    
    # Training loop
    for ep in range(start_epoch, n_epoch):
        print(f'\nEpoch {ep+1}/{n_epoch} - {model_label} (6-hour samples)')
        
        dataset = GPMDataHourly(
            folder_path=data_folder, 
            file_paths=all_file_paths,
            filter_samples=False,
            use_6hour=True,
            skip_init=True
        )

        dataset.hour_groups = filtered_hour_groups
        if dataset.use_6hour:
            dataset.six_hour_samples = filtered_six_hour_samples
        else:
            dataset.hour_keys = filtered_hour_keys
        
        sample_type = "6-hour" if dataset.use_6hour else "1-hour"
        print(f'Using {len(dataset)} filtered {sample_type} samples')
        
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, 
            pin_memory=Config.Training.PIN_MEMORY,
            num_workers=Config.Training.NUM_WORKERS
        )
        
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {ep+1}")
        for batch_idx, x in enumerate(pbar):
            x = x.to(device)
            
            t = torch.randint(0, timesteps, (x.shape[0],), device=device).long()
            
            loss = ddpm.p_losses(model, x, t)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.Training.GRAD_CLIP_NORM)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}', 
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            if batch_idx % Config.Training.SAVE_EVERY_N_BATCHES == 0:
                save_checkpoint(
                    model, optimizer, scheduler, ep, 
                    np.mean(epoch_losses) if epoch_losses else float('inf'), 
                    train_losses, save_dir, is_best=False,
                    learn_variance=learn_variance,
                    schedule=schedule
                )
            
            if batch_idx % Config.Training.FLUSH_EVERY_N_BATCHES == 0:
                torch.cuda.empty_cache()
        
        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            print(f"New best loss: {best_loss:.6f}")
        
        save_checkpoint(
            model, optimizer, scheduler, ep, best_loss, train_losses, save_dir, is_best,
            learn_variance=learn_variance,
            schedule=schedule
        )
        
        # Validation and visualization
        if (ep + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                print(f"Testing with multiple noise levels: {TEST_NOISE_STEPS}")
                
                for sample_idx in range(Config.Visualization.NUM_VIS_SAMPLES):
                    random_idx = random.randint(0, len(dataset) - 1)
                    x_orig = dataset[random_idx].unsqueeze(0).to(device)

                    results_for_sample = []
                    
                    for test_steps in TEST_NOISE_STEPS:
                        actual_test_steps = min(test_steps, timesteps - 1)
                        
                        t_test = torch.full((1,), actual_test_steps, device=device, dtype=torch.long)
                        noise = torch.randn_like(x_orig)
                        x_noisy = ddpm.q_sample(x_orig, t_test, noise)
                        
                        x_denoised = x_noisy.clone()
                        for step in reversed(range(0, actual_test_steps + 1)):
                            t_step = torch.full((1,), step, device=device, dtype=torch.long)
                            x_denoised = ddpm.p_sample(model, x_denoised, t_step, step)

                        x_orig_denorm = custom_denormalize(x_orig)
                        x_noisy_denorm = custom_denormalize(x_noisy)
                        x_denoised_denorm = custom_denormalize(x_denoised)

                        x_orig_denorm = np.clip(x_orig_denorm, 0, x_max)
                        x_noisy_denorm = np.clip(x_noisy_denorm, 0, x_max)
                        x_denoised_denorm = np.clip(x_denoised_denorm, 0, x_max)
                       
                        mse_loss = np.mean((x_denoised_denorm - x_orig_denorm) ** 2)
                        psnr = 10 * np.log10(x_max**2 / mse_loss) if mse_loss != 0 else float('inf')
                        
                        print(f"Sample {sample_idx+1}, t={actual_test_steps}: MSE={mse_loss:.4f}, PSNR={psnr:.2f}dB")

                        results_for_sample.append({
                            'test_steps': actual_test_steps,
                            'x_orig': x_orig_denorm,
                            'x_noisy': x_noisy_denorm,
                            'x_denoised': x_denoised_denorm,
                            'mse': mse_loss,
                            'psnr': psnr
                        })

                    num_steps = len(TEST_NOISE_STEPS)
                    fig, axes = plt.subplots(3, num_steps, figsize=(4*num_steps, 12), 
                                           subplot_kw={'projection': ccrs.PlateCarree()})
                    
                    result_shape = results_for_sample[0]['x_orig'].shape
                    lon_min, lon_max = Config.DataProcessing.LON_RANGE
                    lat_min, lat_max = Config.DataProcessing.LAT_RANGE
                    lon = np.linspace(lon_min, lon_max, result_shape[0])
                    lat = np.linspace(lat_min, lat_max, result_shape[1])
                    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='ij')

                    def make_Rr_cmap(levels):
                        nbin = len(levels) - 1
                        cmap = cm.get_cmap('jet', nbin)
                        norm = mcolors.BoundaryNorm(levels, nbin)
                        return cmap, norm

                    levels_Rr = Config.Visualization.PRECIP_LEVELS
                    cmap_Rr, norm_Rr = make_Rr_cmap(levels_Rr)

                    for step_idx, result in enumerate(results_for_sample):
                        for row_idx, (data_key, title_prefix) in enumerate([
                            ('x_orig', 'Original (6h)'),
                            ('x_noisy', 'Noised'),
                            ('x_denoised', 'Denoised')
                        ]):
                            ax = axes[row_idx, step_idx] if num_steps > 1 else axes[row_idx]
                            
                            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                            ax.add_feature(cfeature.LAND, facecolor='lightgray')
                            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                            ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                                          crs=ccrs.PlateCarree())
                            
                            data_plot = result[data_key] + 1e-6
                            data_plot[data_plot == 0] = np.nan
                            
                            im = ax.pcolormesh(lon_grid, lat_grid, data_plot, 
                                             cmap=cmap_Rr, norm=norm_Rr, alpha=0.5, 
                                             transform=ccrs.PlateCarree())
                            cmap_Rr.set_under('none')
                            
                            if data_key == 'x_denoised':
                                title = f'{title_prefix} MSE:{result["mse"]:.3f} PSNR:{result["psnr"]:.1f}dB'
                            else:
                                title = f'{title_prefix} (t={result["test_steps"]})'
                            
                            ax.set_title(title, fontsize=9)
                            
                            if step_idx == num_steps - 1:
                                cbar = fig.colorbar(im, ax=ax, orientation='vertical', 
                                                   shrink=0.8, ticks=levels_Rr)
                                cbar.set_label('Precipitation (mm)', fontsize='large')

                    plt.tight_layout()
                    
                    config_str = f"{'ImprovedDDPM' if learn_variance else 'StandardDDPM'}_{schedule}_6hour"
                    plt.savefig(
                        os.path.join(save_dir, f"epoch_{ep+1}_sample_{sample_idx+1}_{config_str}.png"), 
                        dpi=Config.Visualization.VIS_DPI, bbox_inches='tight'
                    )
                    plt.close()

            model.train()
    
    # Save training loss curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, alpha=0.7, linewidth=0.5)
    plt.title(f'Training Loss ({model_label}, {schedule} schedule, 6-hour samples)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    window_size = max(len(train_losses) // 100, 10)
    smoothed_losses = []
    for i in range(len(train_losses)):
        start_idx = max(0, i - window_size + 1)
        smoothed_losses.append(np.mean(train_losses[start_idx:i+1]))
    
    plt.subplot(2, 1, 2)
    plt.plot(smoothed_losses, color='red', linewidth=2)
    plt.title(f'Smoothed Training Loss (window={window_size})')
    plt.xlabel('Batch')
    plt.ylabel('Smoothed Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    config_str = f"{'ImprovedDDPM' if learn_variance else 'StandardDDPM'}_{schedule}_6hour"
    loss_plot_path = os.path.join(
        save_dir, 
        f"final_training_loss_{config_str}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(loss_plot_path, dpi=Config.Visualization.VIS_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining complete. Loss curve saved to {loss_plot_path}")
    print(f"Best model loss: {best_loss:.6f}")
    print(f"Config: {model_label}, {schedule} schedule, 6-hour precipitation samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Standard DDPM / Improved DDPM on GPM precipitation data')
    parser.add_argument('--resume', type=str, default=None, 
                       help='path to checkpoint to resume from')
    parser.add_argument('--learn-variance', action='store_true', default=False,
                       help='enable learned variance (Improved DDPM)')
    parser.add_argument('--schedule', type=str, default='cosine', choices=['cosine', 'linear'],
                       help='noise schedule type (cosine or linear)')
    parser.add_argument('--min-precip', type=float, default=None,
                       help=f'minimum precipitation threshold (default: {Config.DataProcessing.MIN_PRECIP_THRESHOLD} mm)')
    parser.add_argument('--no-filter', dest='filter_samples', action='store_false', default=True,
                       help='disable low-precipitation sample filtering')
    
    args = parser.parse_args()
    
    model_name = "Improved DDPM" if args.learn_variance else "Standard DDPM"
    print(f"Starting {model_name} training...")
    print(f"Config: learn_variance={args.learn_variance}, schedule={args.schedule}")
    print(f"Sample filtering: {args.filter_samples} (threshold: {args.min_precip or Config.DataProcessing.MIN_PRECIP_THRESHOLD} mm)")
    
    train_gpm_standard(
        resume_from=args.resume,
        learn_variance=args.learn_variance,
        schedule=args.schedule,
        min_precip_threshold=args.min_precip,
        filter_samples=args.filter_samples
    )