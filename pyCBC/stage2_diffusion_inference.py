import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from tqdm import tqdm
from einops import rearrange
import math
from functools import partial
import warnings
import glob
from datetime import datetime, timedelta
from netCDF4 import Dataset as ncDataset, num2date
warnings.filterwarnings('ignore')

# ===================== Configuration =====================
CASCADE_DIR = '/data/daily/paper1123/zhenduantestallfinal/ensemble_results/ensemble_precipitation'
MODEL_CHECKPOINT = '/data/daily/test1002/best_model_gpm_standard.pth'
DIFFUSION_OUTPUT_DIR = '/data/daily/paper1123/difffinal100/filtered_diffusion_original_results'
GPM_BASE_DIR = '/data/daily'

TARGET_SIZE = (200, 200)
DEVICE = 'cuda:2'
NOISE_STEPS = 100
NUM_DENOISING = 10

MIN_MEAN_PRECIPITATION = 0.001

LON_RANGE = (110, 130)
LAT_RANGE = (20, 40)

# Normalization parameters
x_max = 121
epsilon = 1e-6

# ===================== Utility Functions =====================
def parse_timestamp(timestamp_str):
    """Parses a timestamp string (YYYYMMDDHH) into a datetime object."""
    year = int(timestamp_str[:4])
    month = int(timestamp_str[4:6])
    day = int(timestamp_str[6:8])
    hour = int(timestamp_str[8:10])
    return datetime(year, month, day, hour)

def interpolate_to_target_size(data, target_size=(200, 200)):
    """Bilinearly interpolates data to target_size."""
    if data.shape == target_size:
        return data
    zoom_factors = (target_size[0] / data.shape[0], target_size[1] / data.shape[1])
    interpolated = zoom(data, zoom_factors, order=1)
    return interpolated

# ===================== Normalization =====================
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

# ===================== GPM Data Loading =====================
def find_gpm_files_for_6hours(start_datetime, gpm_base_dir):
    """Finds all GPM half-hourly files within the 6-hour window starting at start_datetime."""
    year = start_datetime.year
    year_dir = os.path.join(gpm_base_dir, f'R{year}')

    if not os.path.exists(year_dir):
        return []

    time_points = [start_datetime + timedelta(minutes=30*i) for i in range(12)]

    gpm_files = []
    for tp in time_points:
        date_str = tp.strftime('%Y%m%d')
        hour_str = tp.strftime('%H')
        minute_str = tp.strftime('%M')
        pattern = f"3B-HHR.MS.MRG.3IMERG.{date_str}-S{hour_str}{minute_str}*"
        matching_files = glob.glob(os.path.join(year_dir, pattern))
        if matching_files:
            gpm_files.append(matching_files[0])

    return sorted(gpm_files)

def load_and_accumulate_gpm_precipitation(file_paths, lon_range=(110, 130), lat_range=(20, 40)):
    """Loads GPM half-hourly files and accumulates them into a 6-hour precipitation field."""
    accumulated_precip = None
    lon = None
    lat = None

    for file_path in file_paths:
        try:
            with ncDataset(file_path, 'r') as ds:
                precip_rate = ds.variables['precipitation'][:].astype(np.float32)

                if lon is None:
                    lat_full = ds.variables['lat'][:].astype(np.float32)
                    lon_full = ds.variables['lon'][:].astype(np.float32)
                    lat_mask = (lat_full >= lat_range[0]) & (lat_full <= lat_range[1])
                    lon_mask = (lon_full >= lon_range[0]) & (lon_full <= lon_range[1])
                    lat = lat_full[lat_mask]
                    lon = lon_full[lon_mask]

                if len(precip_rate.shape) == 3:
                    precip_rate = precip_rate[0]

                lat_full = ds.variables['lat'][:].astype(np.float32)
                lon_full = ds.variables['lon'][:].astype(np.float32)
                lat_mask = (lat_full >= lat_range[0]) & (lat_full <= lat_range[1])
                lon_mask = (lon_full >= lon_range[0]) & (lon_full <= lon_range[1])
                precip_rate = precip_rate[lon_mask, :][:, lat_mask]

                if accumulated_precip is None:
                    accumulated_precip = precip_rate * 0.5
                else:
                    accumulated_precip += precip_rate * 0.5

        except Exception as e:
            print(f"  Failed to read {os.path.basename(file_path)}: {e}")
            continue

    if accumulated_precip is not None:
        accumulated_precip = accumulated_precip.T

    return accumulated_precip, lon, lat

def load_ground_truth_from_gpm(sample_name, gpm_base_dir, lon_range=(110, 130), lat_range=(20, 40)):
    """Loads 6-hour accumulated ground-truth precipitation from GPM files."""
    try:
        start_datetime = parse_timestamp(sample_name)
        gpm_files = find_gpm_files_for_6hours(start_datetime, gpm_base_dir)

        if len(gpm_files) == 0:
            return None

        truth, lon, lat = load_and_accumulate_gpm_precipitation(gpm_files, lon_range, lat_range)

        if truth is None:
            return None

        return truth

    except Exception as e:
        return None

# ===================== Sample Filtering =====================
def find_all_cascade_samples(cascade_dir):
    """Returns sorted list of all cascade forecast sample names found in cascade_dir."""
    if not os.path.exists(cascade_dir):
        print(f"Error: cascade directory not found: {cascade_dir}")
        return []

    files = [f for f in os.listdir(cascade_dir) if f.endswith('.npy') and f.startswith('ensemble_v2_')]

    sample_names = []
    for filename in files:
        sample_name = filename.replace('ensemble_v2_', '').replace('.npy', '')
        sample_names.append(sample_name)

    return sorted(sample_names)

def filter_samples_by_precipitation(cascade_samples, gpm_base_dir, min_mean_precip=3.0):
    """Retains only samples whose GPM ground-truth mean precipitation meets the threshold."""
    print(f"\nFiltering samples (mean precipitation >= {min_mean_precip} mm)...")

    filtered_samples = []
    sample_precip_stats = {}

    for sample_name in tqdm(cascade_samples, desc="Filtering samples"):
        try:
            ground_truth = load_ground_truth_from_gpm(sample_name, gpm_base_dir, LON_RANGE, LAT_RANGE)

            if ground_truth is None:
                continue

            mean_precip = np.mean(ground_truth)
            max_precip = np.max(ground_truth)

            sample_precip_stats[sample_name] = {
                'mean': mean_precip,
                'max': max_precip,
                'passed': mean_precip >= min_mean_precip
            }

            if mean_precip >= min_mean_precip:
                filtered_samples.append(sample_name)

        except Exception as e:
            continue

    print(f"\nFiltering results:")
    print(f"  Total samples:    {len(cascade_samples)}")
    print(f"  Filtered samples: {len(filtered_samples)}")
    print(f"  Retention rate:   {len(filtered_samples)/len(cascade_samples)*100:.1f}%")

    if sample_precip_stats:
        all_means = [stats['mean'] for stats in sample_precip_stats.values()]
        print(f"\nAll samples - precipitation stats:")
        print(f"  Mean range: {np.min(all_means):.2f} - {np.max(all_means):.2f} mm")
        print(f"  Mean:       {np.mean(all_means):.2f} mm")

        passed_means = [stats['mean'] for stats in sample_precip_stats.values() if stats['passed']]
        if passed_means:
            print(f"\nFiltered samples - precipitation stats:")
            print(f"  Mean range: {np.min(passed_means):.2f} - {np.max(passed_means):.2f} mm")
            print(f"  Mean:       {np.mean(passed_means):.2f} mm")

    return filtered_samples, sample_precip_stats

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
    """Standard UNet for diffusion post-processing (no spatial encoding)."""
    def __init__(self, dim_in, dim_out, c, c_mults=(1, 2, 4, 8),
                 resnet_block_groups=4, scale=[2,2,2], learn_variance=False):
        super().__init__()
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

# ===================== DDPM =====================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DDPM:
    def __init__(self, timesteps=1000, device='cuda', learn_variance=False, schedule='cosine'):
        self.device = device
        self.timesteps = timesteps
        self.learn_variance = learn_variance

        if schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps).to(device)
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

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        device = t.device
        a = a.to(device)
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def model_predictions(self, model, x_t, t):
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
        pred_noise, pred_x0, model_variance = self.model_predictions(model, x_t, t)

        model_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0 +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        return model_mean, model_variance, pred_x0

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        model_mean, model_variance, pred_x0 = self.p_mean_variance(model, x, t)

        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(model_variance) * noise

    @torch.no_grad()
    def denoise_from_t(self, model, x_noisy, start_t):
        x = x_noisy.clone()
        batch_size = x.shape[0]

        for i in reversed(range(0, start_t + 1)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)

        return x

# ===================== Diffusion Post-Processing =====================
def process_filtered_samples_with_diffusion(filtered_samples, cascade_dir, output_dir,
                                            model, ddpm, num_denoising=10):
    """Applies SDEdit-style diffusion post-processing to each filtered cascade forecast sample."""
    print("\n" + "="*100)
    print("Starting diffusion post-processing")
    print("="*100)

    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0

    with torch.no_grad():
        for sample_name in tqdm(filtered_samples, desc="Diffusion processing"):
            try:
                cascade_file = f"ensemble_v2_{sample_name}.npy"
                cascade_path = os.path.join(cascade_dir, cascade_file)

                if not os.path.exists(cascade_path):
                    print(f"\n  Warning: cascade file not found: {cascade_file}")
                    continue

                # Load and preprocess cascade prediction
                pred_data = np.load(cascade_path)
                pred_interpolated = interpolate_to_target_size(pred_data, TARGET_SIZE)
                pred_normalized = custom_normalize(pred_interpolated)
                pred_tensor = torch.from_numpy(pred_normalized[np.newaxis, np.newaxis, :, :]).float().to(DEVICE)

                # Generate num_denoising ensemble members via SDEdit
                for denoise_idx in range(num_denoising):
                    t_noise = torch.full((1,), NOISE_STEPS, device=DEVICE, dtype=torch.long)
                    noise = torch.randn_like(pred_tensor)
                    x_noisy = ddpm.q_sample(pred_tensor, t_noise, noise)

                    x_denoised = ddpm.denoise_from_t(model, x_noisy, NOISE_STEPS)

                    x_denoised_denorm = custom_denormalize(x_denoised)
                    x_denoised_denorm = np.clip(x_denoised_denorm, 0, x_max)

                    output_filename = f"ensemble_v2_{sample_name}_denoised_{denoise_idx+1}.npy"
                    output_path = os.path.join(output_dir, output_filename)
                    np.save(output_path, x_denoised_denorm)

                processed_count += 1

                # Periodic VRAM cache flush
                if processed_count % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n  Error processing sample {sample_name}: {e}")
                continue

    print(f"\nProcessing complete. {processed_count} samples processed.")
    print(f"{num_denoising} denoised ensemble members generated per sample.")
    print(f"Output directory: {output_dir}")

    return processed_count

# ===================== Ensemble Aggregation =====================
def calculate_ensemble_mean(ensemble):
    return np.mean(ensemble, axis=0)

def calculate_ensemble_median(ensemble):
    return np.median(ensemble, axis=0)

def calculate_ensemble_p80(ensemble):
    return np.percentile(ensemble, 80, axis=0)

def calculate_ensemble_p90(ensemble):
    return np.percentile(ensemble, 90, axis=0)

def calculate_probability_matched_mean(ensemble):
    """Probability-matched mean: applies the full-ensemble value distribution to the ensemble mean field."""
    ensemble_mean = np.mean(ensemble, axis=0)
    mean_flat = ensemble_mean.flatten()
    mean_sort_indices = np.argsort(mean_flat)

    all_values = ensemble.flatten()
    all_values_sorted = np.sort(all_values)

    n_points = len(mean_flat)
    sampled_values = np.zeros(n_points)
    for i, idx in enumerate(mean_sort_indices):
        position_in_all = int((i / n_points) * len(all_values_sorted))
        position_in_all = min(position_in_all, len(all_values_sorted) - 1)
        sampled_values[idx] = all_values_sorted[position_in_all]

    pmm = sampled_values.reshape(ensemble_mean.shape)
    return pmm

# ===================== Data Loading =====================
def load_diffusion_ensemble(diffusion_dir, sample_name, num_denoising=10):
    """Loads all diffusion ensemble members for a given sample."""
    ensemble = []
    for i in range(1, num_denoising + 1):
        filename = f"ensemble_v2_{sample_name}_denoised_{i}.npy"
        filepath = os.path.join(diffusion_dir, filename)

        if os.path.exists(filepath):
            data = np.load(filepath)
            data = np.flip(data, axis=0)  # fix vertical flip
            ensemble.append(data)

    return np.array(ensemble) if len(ensemble) == num_denoising else None

def load_cascade_prediction(cascade_dir, sample_name):
    """Loads the cascade forecast .npy file for a given sample."""
    filename = f"ensemble_v2_{sample_name}.npy"
    filepath = os.path.join(cascade_dir, filename)

    if os.path.exists(filepath):
        data = np.load(filepath)
        data = np.flip(data, axis=0)  # fix vertical flip
        return data
    return None

# ===================== Main =====================
def main():
    print("=" * 100)
    print("Precipitation post-processing pipeline: Filter -> Diffusion")
    print("Standard UNet diffusion model")
    print("=" * 100)
    print(f"Cascade directory:   {CASCADE_DIR}")
    print(f"Model checkpoint:    {MODEL_CHECKPOINT}")
    print(f"Output directory:    {DIFFUSION_OUTPUT_DIR}")
    print(f"GPM data directory:  {GPM_BASE_DIR}")
    print(f"Filter threshold:    mean precipitation >= {MIN_MEAN_PRECIPITATION} mm")
    print("=" * 100)

    # ========== Stage 1: Sample Filtering ==========
    print("\n" + "="*100)
    print("Stage 1: Sample Filtering")
    print("="*100)

    print("\nSearching for cascade forecast samples...")
    all_cascade_samples = find_all_cascade_samples(CASCADE_DIR)

    if not all_cascade_samples:
        print("Error: no cascade forecast samples found!")
        return

    print(f"Found {len(all_cascade_samples)} cascade forecast samples")

    print("\nFiltering by precipitation threshold...")
    filtered_samples, sample_precip_stats = filter_samples_by_precipitation(
        all_cascade_samples, GPM_BASE_DIR, MIN_MEAN_PRECIPITATION
    )

    if not filtered_samples:
        print("Error: no samples passed the filter!")
        return

    print(f"\nFiltering complete. {len(filtered_samples)} samples retained.")

    # ========== Stage 2: Diffusion Post-Processing ==========
    print("\n" + "="*100)
    print("Stage 2: Diffusion Post-Processing (Standard UNet)")
    print("="*100)

    print("\nLoading diffusion model...")
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    learn_variance = checkpoint.get('learn_variance', False)
    schedule = checkpoint.get('schedule', 'cosine')

    print(f"Model config:")
    print(f"  Type:            Standard UNet (no spatial encoding)")
    print(f"  learn_variance:  {learn_variance}")
    print(f"  schedule:        {schedule}")

    model = UNet(
        dim_in=1, dim_out=1, c=128, c_mults=(1, 2, 4, 8),
        resnet_block_groups=4, scale=[2, 2, 2],
        learn_variance=learn_variance
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    ddpm = DDPM(
        timesteps=1000,
        device=DEVICE,
        learn_variance=learn_variance,
        schedule=schedule
    )

    print("\nStarting diffusion post-processing...")
    processed_count = process_filtered_samples_with_diffusion(
        filtered_samples, CASCADE_DIR, DIFFUSION_OUTPUT_DIR,
        model, ddpm, NUM_DENOISING
    )

    if processed_count == 0:
        print("Error: diffusion processing produced no output!")
        return

    print("\n" + "=" * 100)
    print("Pipeline complete.")
    print("=" * 100)
    print(f"\nSample statistics:")
    print(f"  Total samples:     {len(all_cascade_samples)}")
    print(f"  Filtered samples:  {len(filtered_samples)}")
    print(f"  Processed samples: {processed_count}")
    print(f"\nDiffusion output: {DIFFUSION_OUTPUT_DIR}")
    print("=" * 100)

if __name__ == "__main__":
    main()
