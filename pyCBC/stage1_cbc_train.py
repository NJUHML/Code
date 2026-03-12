
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import cpu_count
import re
import torch.nn.functional as F
import json
import pickle
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Unified configuration class managing all hyperparameters and paths."""
    
    # ==================== Data Paths ====================
    class DataPaths:
        ERA5_BASE_PATH = '/data/daily/Diffusion1106/era5_processed_6h_multilayer/'
        PRECIP_6H_DIR = '/data/daily/Diffusion1106/precipitation_6h/'
        
        # Metadata and normalization stats cache
        METADATA_PATH = '/data/daily/paper1123/finalzhenduan/direct_load_meta.pkl'
        FILTER_CACHE_PATH = '/data/daily/paper1123/finalzhenduan/precipitation_filter_cache_6h.pkl'
        
        # Normalization statistics
        STATS_SAVE_PATH = '/data/daily/paper1123/finalzhenduan/normalization_stats_filteredmore.pkl'
        
        FILE_INDEX_CACHE_PATH = '/data/daily/paper1123/finalzhenduan/file_index_cache.pkl'
        
        # Model checkpoint directory
        MODEL_DIR = '/data/daily/paper1123/finalzhenduan/finalzhenduanmodel1/'
    
    # ==================== ERA5 Variables ====================
    class ERA5Variables:
        UPPER_VARS = ['u', 'v', 't', 'z', 'rh','vertical_velocity']
        PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] 
        SURFACE_VARS = ['u10', 'v10', 't2m', 'msl']
        
        N_UPPER_CHANNELS = len(UPPER_VARS) * len(PRESSURE_LEVELS)  # 6 vars x 13 levels = 78
        N_SURFACE_CHANNELS = len(SURFACE_VARS)  # 4
        N_LATLON_CHANNELS = 2  # lat, lon
        N_TOTAL_CHANNELS = N_UPPER_CHANNELS + N_SURFACE_CHANNELS + N_LATLON_CHANNELS  # 84
    
    # ==================== Data Processing ====================
    class DataProcessing:
        # Full spatial domain (retains boundary context)
        LON_RANGE = (108, 132)
        LAT_RANGE = (18, 42)
        
        # Core evaluation domain (loss computed here only)
        LOSS_LON_RANGE = (110, 130)
        LOSS_LAT_RANGE = (20, 40)
        
        SAMPLE_FILTER_THRESHOLD = 1.0  # minimum domain-mean precipitation for sample inclusion (mm/6h)
        
        # Precipitation thresholds for the 15 binary classifiers
        TRAINING_THRESHOLDS = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 
                               40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        
        FORCE_RECOMPUTE_METADATA = False
        FORCE_RECOMPUTE_FILTER = False
        FORCE_RECOMPUTE_STATS = False
        FORCE_REBUILD_FILE_INDEX = False
        
        NORM_MAX_SAMPLES = 2000  # max samples used for computing normalization statistics
    
    # ==================== Data Split ====================
    class DataSplit:
        TRAIN_YEARS = [2013, 2014, 2015, 2017, 2018, 2019, 2021, 2022]
        VAL_YEARS = [2016, 2020]
    
    # ==================== Model Architecture ====================
    class Model:
        IN_CHANNELS = 84
        GUIYIHUA = IN_CHANNELS-2
        BASE_CHANNELS = 64
        CHANNEL_MULTS = (1, 2, 4, 8)
        DROPOUT_RATE = 0.05
        USE_DROPOUT_SCHEDULE = True
        DROPOUT_MIN_RATE = 0.01
        DROPOUT_DECAY_EPOCHS = 10
    
    # ==================== Training Hyperparameters ====================
    class Training:
        BATCH_SIZE = 12
        NUM_EPOCHS = 150
        INITIAL_LR = 0.0008
        WEIGHT_DECAY = 0.015
        GRAD_CLIP_NORM = 1.0
        EARLY_STOPPING_PATIENCE = 3  # CSI can fluctuate; use larger patience
        LR_SCHEDULER_FACTOR = 0.5
        LR_SCHEDULER_PATIENCE = 6
        LR_SCHEDULER_MIN_LR = 1e-6
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        NUM_WORKERS = min(6, cpu_count())
        PIN_MEMORY = True
    
    # ==================== Loss Function ====================
    class Loss:
        LOSS_TYPE = 'combined'
        BCE_WEIGHT = 0.8
        DICE_WEIGHT = 0.2
        DICE_SMOOTH = 1e-6
        CLASS_WEIGHT_SAMPLE_SIZE = 1000
    
    # ==================== Threshold Search ====================
    class Threshold:
        THRESHOLD_MIN = 0.5
        THRESHOLD_MAX = 1.0
        NUM_THRESHOLDS = 250


# ============================================================================
# Core Region Mask Generator
# ============================================================================

class CoreRegionMaskGenerator:
    """Generates a spatial mask for the core evaluation domain, restricting loss computation."""
    
    def __init__(self, full_lon_range=None, full_lat_range=None,
                 core_lon_range=None, core_lat_range=None, grid_shape=(97, 97)):
        """
        Args:
            full_lon_range: longitude range of full domain (108, 132)
            full_lat_range: latitude range of full domain (18, 42)
            core_lon_range: longitude range of core domain (110, 130)
            core_lat_range: latitude range of core domain (20, 40)
            grid_shape: spatial grid dimensions
        """
        if full_lon_range is None:
            full_lon_range = Config.DataProcessing.LON_RANGE
        if full_lat_range is None:
            full_lat_range = Config.DataProcessing.LAT_RANGE
        if core_lon_range is None:
            core_lon_range = Config.DataProcessing.LOSS_LON_RANGE
        if core_lat_range is None:
            core_lat_range = Config.DataProcessing.LOSS_LAT_RANGE
        
        self.full_lon_range = full_lon_range
        self.full_lat_range = full_lat_range
        self.core_lon_range = core_lon_range
        self.core_lat_range = core_lat_range
        self.grid_shape = grid_shape
        
        # Generate mask
        self.mask = self._generate_mask()
        
        # Fraction of grid points inside core domain
        self.core_ratio = self.mask.sum() / self.mask.numel()
        
        print(f"Core region mask initialized:")
        print(f"  Full domain: lon{full_lon_range}, lat{full_lat_range}")
        print(f"  Core domain: lon{core_lon_range}, lat{core_lat_range}")
        print(f"  Grid shape: {grid_shape}")
        print(f"  Core coverage: {self.core_ratio*100:.2f}%")
    
    def _generate_mask(self):
        """Returns a float mask (1 = core region, 0 = boundary padding)."""
        n_lat, n_lon = self.grid_shape
        
        # Build lat/lon coordinate grids
        lons = np.linspace(self.full_lon_range[0], self.full_lon_range[1], n_lon)
        lats = np.linspace(self.full_lat_range[0], self.full_lat_range[1], n_lat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Boolean mask for points inside core domain
        lon_mask = (lon_grid >= self.core_lon_range[0]) & (lon_grid <= self.core_lon_range[1])
        lat_mask = (lat_grid >= self.core_lat_range[0]) & (lat_grid <= self.core_lat_range[1])
        
        mask = lon_mask & lat_mask
        
        return torch.from_numpy(mask.astype(np.float32))
    
    def get_mask(self, device=None):
        """Returns the mask tensor, optionally moved to the specified device."""
        if device is not None:
            return self.mask.to(device)
        return self.mask
    
    def apply_mask(self, tensor, fill_value=0.0):
        """Zeros out (or fills) non-core regions of a tensor."""
        return tensor * self.mask + fill_value * (1 - self.mask)


# ============================================================================
# Utility Classes
# ============================================================================

class LatLonGridGenerator:
    """Generates normalized lat/lon coordinate grids as additional input channels."""
    
    def __init__(self, lon_range=None, lat_range=None, grid_shape=(97, 97)):
        if lon_range is None:
            lon_range = Config.DataProcessing.LON_RANGE
        if lat_range is None:
            lat_range = Config.DataProcessing.LAT_RANGE
        
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.grid_shape = grid_shape
        
        lat_grid, lon_grid = self._generate_grids()
        self.lat_tensor = torch.from_numpy(lat_grid).float()
        self.lon_tensor = torch.from_numpy(lon_grid).float()
        
        print(f"Lat/lon grid generator initialized: {grid_shape}")
    
    def _generate_grids(self):
        n_lat, n_lon = self.grid_shape
        lons = np.linspace(self.lon_range[0], self.lon_range[1], n_lon)
        lats = np.linspace(self.lat_range[0], self.lat_range[1], n_lat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        lon_normalized = 2 * (lon_grid - self.lon_range[0]) / (self.lon_range[1] - self.lon_range[0]) - 1
        lat_normalized = 2 * (lat_grid - self.lat_range[0]) / (self.lat_range[1] - self.lat_range[0]) - 1
        
        return lat_normalized, lon_normalized
    
    def get_grids(self):
        return torch.stack([self.lat_tensor, self.lon_tensor], dim=0)


class CustomInputNormalize:
    """Per-channel normalization for 82-channel ERA5 input."""
    
    def __init__(self, mean, std):
        self.mean = mean.unsqueeze(-1).unsqueeze(-1)
        self.std = std.unsqueeze(-1).unsqueeze(-1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / (self.std + 1e-8)


class DropoutScheduler:
    """Linearly anneals dropout rate from initial_rate to min_rate over decay_epochs."""
    
    def __init__(self, model, initial_rate=None, min_rate=None, decay_epochs=None):
        self.model = model
        self.initial_rate = initial_rate or Config.Model.DROPOUT_RATE
        self.min_rate = min_rate or Config.Model.DROPOUT_MIN_RATE
        self.decay_epochs = decay_epochs or Config.Model.DROPOUT_DECAY_EPOCHS
    
    def step(self, epoch):
        if epoch < self.decay_epochs:
            new_rate = self.initial_rate - (self.initial_rate - self.min_rate) * (epoch / self.decay_epochs)
        else:
            new_rate = self.min_rate
        
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.p = new_rate
        
        return new_rate


# ============================================================================
# File Index Builder
# ============================================================================

class FileIndexBuilder:
    """Builds a timestamp-keyed file index for ERA5 variables to avoid repeated directory scans."""
    
    @staticmethod
    def build_file_index(era5_base, var_dirs, cache_path=None, force_rebuild=False):
        """Scans all variable directories and returns a nested dict: {timestamp: {var_idx: filepath}}."""
        
        # Load from cache if available
        if cache_path and os.path.exists(cache_path) and not force_rebuild:
            print(f"\nLoading file index from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if cached_data.get('era5_base') == era5_base and \
                   len(cached_data.get('file_index', {})) > 0:
                    print(f"File index loaded: {len(cached_data['file_index'])} timestamps, {len(var_dirs)} variables")
                    return cached_data['file_index']
                else:
                    print("Cache config mismatch, rebuilding index...")
            except Exception as e:
                print(f"Cache load failed: {e}, rebuilding index...")
        
        print("\n" + "=" * 80)
        print("Building ERA5 file index (cached after first run)")
        print("=" * 80)
        print(f"ERA5 base path: {era5_base}")
        print(f"Number of variables: {len(var_dirs)}")
        
        file_index = {}
        total_files = 0
        
        for var_idx, var_dir in enumerate(tqdm(var_dirs, desc="Scanning variable directories")):
            if not os.path.exists(var_dir):
                print(f"Warning: directory not found - {var_dir}")
                continue
            
            try:
                all_files = os.listdir(var_dir)
                var_files = [f for f in all_files if f.endswith('.npy')]
                
                for fname in var_files:
                    match = re.search(r'_?(\d{10})\.npy$', fname)
                    if match:
                        timestamp = match.group(1)
                        
                        if timestamp not in file_index:
                            file_index[timestamp] = {}
                        
                        file_index[timestamp][var_idx] = os.path.join(var_dir, fname)
                        total_files += 1
                
            except Exception as e:
                print(f"Error scanning {var_dir}: {e}")
                continue
        
        print(f"\nFile index built: {len(file_index)} timestamps, {total_files} total files")
        
        # Save cache
        if cache_path:
            cache_data = {
                'file_index': file_index,
                'era5_base': era5_base,
                'var_dirs': var_dirs,
                'build_time': datetime.now().isoformat(),
                'total_files': total_files
            }
            
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"File index cached: {cache_path}")
        
        return file_index


# ============================================================================
# CSI Metric Functions
# ============================================================================

def compute_csi(outputs, targets, threshold, mask=None):
    """
    Computes the Critical Success Index (Threat Score): CSI = TP / (TP + FP + FN).

    CSI ignores true negatives and focuses on the quality of precipitation event forecasts.

    Args:
        outputs: model output probabilities
        targets: binary ground truth
        threshold: decision threshold for binarizing outputs
        mask: optional spatial mask; if provided, only masked grid points are evaluated

    Returns:
        csi: CSI score in [0, 1]
        tp, fp, fn: contingency table elements
    """
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Apply spatial mask
    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
        if mask_np.shape != outputs_np.shape:
            mask_np = np.broadcast_to(mask_np, outputs_np.shape)
    else:
        mask_np = np.ones_like(outputs_np)
    
    # Binarize predictions
    pred = (outputs_np > threshold).astype(np.float32)
    
    # Restrict to masked region
    pred_masked = pred[mask_np > 0.5]
    target_masked = targets_np[mask_np > 0.5]
    
    # Contingency table
    tp = np.sum((pred_masked == 1) & (target_masked == 1))
    fp = np.sum((pred_masked == 1) & (target_masked == 0))
    fn = np.sum((pred_masked == 0) & (target_masked == 1))
    
    # CSI
    csi = tp / (tp + fp + fn + 1e-8)
    
    return csi, tp, fp, fn


def find_optimal_threshold_csi(outputs, targets, threshold_range=None, num_thresholds=None, mask=None):
    """
    Searches for the decision threshold that maximizes CSI.

    Args:
        outputs: model output probabilities
        targets: binary ground truth
        threshold_range: (min, max) search bounds
        num_thresholds: number of candidate thresholds
        mask: optional spatial mask

    Returns:
        best_threshold: threshold achieving highest CSI
        best_csi: corresponding CSI value
    """
    if threshold_range is None:
        threshold_range = (Config.Threshold.THRESHOLD_MIN, Config.Threshold.THRESHOLD_MAX)
    if num_thresholds is None:
        num_thresholds = Config.Threshold.NUM_THRESHOLDS
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    best_csi = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        csi, _, _, _ = compute_csi(outputs, targets, threshold, mask)
        
        if csi > best_csi:
            best_csi = csi
            best_threshold = threshold
    
    return best_threshold, best_csi


# ============================================================================
# Helper Functions
# ============================================================================

def parse_timestamp_from_filename(filename):
    """Extracts a 10-digit timestamp from a filename and returns a datetime object."""
    match = re.search(r'_?(\d{10})\.npy$', filename)
    if match:
        timestamp_str = match.group(1)
        year = int(timestamp_str[:4])
        month = int(timestamp_str[4:6])
        day = int(timestamp_str[6:8])
        hour = int(timestamp_str[8:10])
        return datetime(year, month, day, hour)
    return None


# ============================================================================
# Data Processing Functions
# ============================================================================

def build_metadata_from_existing_files(precip_dir=None, era5_base=None, 
                                       metadata_path=None, force_recompute=None):
    """Scans precipitation and ERA5 directories to build a sample list keyed by matched timestamps."""
    if precip_dir is None:
        precip_dir = Config.DataPaths.PRECIP_6H_DIR
    if era5_base is None:
        era5_base = Config.DataPaths.ERA5_BASE_PATH
    if metadata_path is None:
        metadata_path = Config.DataPaths.METADATA_PATH
    if force_recompute is None:
        force_recompute = Config.DataProcessing.FORCE_RECOMPUTE_METADATA
    
    # Load from cache
    if os.path.exists(metadata_path) and not force_recompute:
        print(f"Loading metadata from cache: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"Loaded {len(metadata['samples'])} samples")
        return metadata
    
    print("\n=== Building metadata from existing files ===")
    print(f"Precipitation dir: {precip_dir}")
    print(f"ERA5 base dir: {era5_base}")
    
    # Enumerate precipitation files
    precip_files = sorted([f for f in os.listdir(precip_dir) if f.endswith('.npy')])
    print(f"Found {len(precip_files)} precipitation files")
    
    # Locate a representative ERA5 variable directory for timestamp scanning
    sample_var_dir = os.path.join(era5_base, 't2m')
    if not os.path.exists(sample_var_dir):
        for test_var in ['u10', 'v10', 'msl', 'u50hpa']:
            sample_var_dir = os.path.join(era5_base, test_var)
            if os.path.exists(sample_var_dir):
                break
        else:
            raise FileNotFoundError(f"ERA5 directory not found, check path: {era5_base}")
    
    era5_sample_files = sorted([f for f in os.listdir(sample_var_dir) if f.endswith('.npy')])
    print(f"ERA5 sample files: {len(era5_sample_files)}")
    
    # Build timestamp-to-filename maps
    precip_time_map = {}
    for fname in precip_files:
        ts = parse_timestamp_from_filename(fname)
        if ts:
            precip_time_map[ts] = fname
    
    era5_time_map = {}
    for fname in era5_sample_files:
        ts = parse_timestamp_from_filename(fname)
        if ts:
            era5_time_map[ts] = fname
    
    # Intersect timestamps
    common_times = set(precip_time_map.keys()) & set(era5_time_map.keys())
    common_times = sorted(common_times)
    
    print(f"Matched timestamps: {len(common_times)}")
    
    if len(common_times) == 0:
        raise ValueError("No matching timestamps between precipitation and ERA5 data")
    
    # Build sample list
    samples = []
    for ts in common_times:
        samples.append({
            'timestamp': ts.strftime('%Y%m%d%H'),
            'datetime': ts.isoformat(),
            'precip_file': precip_time_map[ts],
            'year': ts.year
        })
    
    # Infer grid shape from first precipitation file
    sample_precip = np.load(os.path.join(precip_dir, precip_files[0]))
    grid_shape = sample_precip.shape
    
    metadata = {
        'samples': samples,
        'n_samples': len(samples),
        'grid_shape': grid_shape,
        'precip_dir': precip_dir,
        'era5_base': era5_base,
        'n_channels': 46,
        'time_range': {
            'start': common_times[0].isoformat(),
            'end': common_times[-1].isoformat()
        }
    }
    
    # Save metadata
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Metadata saved: {metadata_path}")
    print(f"Total samples: {len(samples)}")
    print(f"Grid shape: {grid_shape}")
    
    return metadata


def filter_samples_by_precipitation(metadata, filter_cache_path=None, 
                                   force_recompute=None, precipitation_threshold=None):
    """Filters the sample list to retain only timesteps where domain-mean precipitation exceeds the threshold."""
    if filter_cache_path is None:
        filter_cache_path = Config.DataPaths.FILTER_CACHE_PATH
    if force_recompute is None:
        force_recompute = Config.DataProcessing.FORCE_RECOMPUTE_FILTER
    if precipitation_threshold is None:
        precipitation_threshold = Config.DataProcessing.SAMPLE_FILTER_THRESHOLD
    
    # Load from cache
    if os.path.exists(filter_cache_path) and not force_recompute:
        print(f"Loading precipitation filter from cache: {filter_cache_path}")
        with open(filter_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        cached_threshold = cache_data.get('threshold', None)
        if cached_threshold != precipitation_threshold:
            print(f"Warning: cached threshold {cached_threshold} != current threshold {precipitation_threshold}, recomputing...")
        else:
            max_idx = max(cache_data['valid_indices']) if cache_data['valid_indices'] else 0
            if max_idx >= len(metadata['samples']):
                print(f"Warning: cached indices out of range, recomputing...")
            else:
                print(f"Loaded filter: {len(cache_data['valid_indices'])} valid samples")
                return cache_data['valid_indices'], cache_data['precipitation_stats']
    
    print(f"\n=== Filtering samples by 6-hour precipitation ===")
    print(f"Threshold: {precipitation_threshold} mm/6h")
    
    samples = metadata['samples']
    precip_dir = metadata['precip_dir']
    total_samples = len(samples)
    
    valid_indices = []
    precipitation_values = []
    
    for idx, sample in enumerate(tqdm(samples, desc="Filtering samples")):
        try:
            precip_path = os.path.join(precip_dir, sample['precip_file'])
            precip_data = np.load(precip_path)
            precip_data = np.nan_to_num(precip_data, nan=0.0, posinf=0.0, neginf=0.0)
            mean_precip = np.mean(precip_data)
            precipitation_values.append(mean_precip)
            
            if mean_precip >= precipitation_threshold:
                valid_indices.append(idx)
                
        except Exception as e:
            print(f"Error processing {sample['precip_file']}: {e}")
            precipitation_values.append(0.0)
            continue
    
    precipitation_stats = {
        'max_precip': np.max(precipitation_values) if precipitation_values else 0.0,
        'min_precip': np.min(precipitation_values) if precipitation_values else 0.0,
        'mean_precip': np.mean(precipitation_values) if precipitation_values else 0.0,
        'std_precip': np.std(precipitation_values) if precipitation_values else 0.0,
        'threshold': precipitation_threshold,
        'valid_count': len(valid_indices),
        'total_count': total_samples,
        'filter_ratio': len(valid_indices) / total_samples if total_samples > 0 else 0.0
    }
    
    print(f"\n=== Filter results ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {len(valid_indices)}")
    print(f"Filter ratio: {len(valid_indices)/total_samples*100:.2f}%")
    
    # Save cache
    if filter_cache_path:
        cache_data = {
            'valid_indices': valid_indices,
            'precipitation_stats': precipitation_stats,
            'total_samples': total_samples,
            'threshold': precipitation_threshold
        }
        os.makedirs(os.path.dirname(filter_cache_path), exist_ok=True)
        with open(filter_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Filter results cached: {filter_cache_path}")
    
    return valid_indices, precipitation_stats


def compute_normalization_stats_from_filtered(metadata, train_indices, file_index,
                                              stats_save_path=None, force_recompute=None):
    """
    Computes per-channel mean and std from filtered training samples using Welford's online algorithm.

    Normalization statistics are derived exclusively from precipitation-filtered training samples,
    ensuring they reflect the distribution of high-precipitation events.

    Args:
        metadata: metadata dictionary
        train_indices: list of sample indices (relative to metadata['samples'])
        file_index: prebuilt file index dict {timestamp: {var_idx: filepath}}
        stats_save_path: path to save/load cached statistics
        force_recompute: if True, recompute even if cache exists

    Returns:
        stats: dict containing 'mean' and 'std' tensors
    """
    if stats_save_path is None:
        stats_save_path = Config.DataPaths.STATS_SAVE_PATH
    if force_recompute is None:
        force_recompute = Config.DataProcessing.FORCE_RECOMPUTE_STATS
    
    # Load from cache if channel count matches
    if os.path.exists(stats_save_path) and not force_recompute:
        print(f"Loading normalization stats from cache: {stats_save_path}")
        with open(stats_save_path, 'rb') as f:
            stats = pickle.load(f)
        
        if stats.get('n_channels') == Config.Model.GUIYIHUA:
            print(f"Normalization stats loaded successfully")
            return stats
        else:
            print(f"Warning: cached channel count mismatch, recomputing...")
    
    print("\n" + "=" * 80)
    print("Computing normalization stats from filtered training samples (Welford online algorithm)")
    print("=" * 80)
    print(f"Training samples: {len(train_indices)}")
    
    # Filter out-of-range indices
    samples = metadata['samples']
    valid_train_indices = [idx for idx in train_indices if idx < len(samples)]
    
    if len(valid_train_indices) < len(train_indices):
        print(f"Warning: {len(train_indices) - len(valid_train_indices)} indices out of range, filtered out")
    
    # Subsample if needed to limit compute
    max_samples = Config.DataProcessing.NORM_MAX_SAMPLES
    if len(valid_train_indices) > max_samples:
        print(f"Subsampling {max_samples} samples for efficiency")
        sample_indices = np.random.choice(valid_train_indices, max_samples, replace=False)
    else:
        sample_indices = valid_train_indices
    
    print(f"Samples used: {len(sample_indices)}")
    
    # Build channel name list for logging
    n_channels = Config.Model.GUIYIHUA
    channel_names = []
    for var in Config.ERA5Variables.UPPER_VARS:
        for level in Config.ERA5Variables.PRESSURE_LEVELS:
            channel_names.append(f"{var}_{level}hPa")
    for var in Config.ERA5Variables.SURFACE_VARS:
        channel_names.append(var)
    
    # Build variable directory list (matches channel ordering)
    var_dirs = []
    for var in Config.ERA5Variables.UPPER_VARS:
        for level in Config.ERA5Variables.PRESSURE_LEVELS:
            dir_name = f"{var}{level}hpa"
            var_dirs.append(os.path.join(metadata['era5_base'], dir_name))
    for var in Config.ERA5Variables.SURFACE_VARS:
        var_dirs.append(os.path.join(metadata['era5_base'], var))
    
    # Welford's online algorithm: accumulate count, mean, and M2 per channel
    channel_count = np.zeros(n_channels, dtype=np.int64)
    channel_mean = np.zeros(n_channels, dtype=np.float64)
    channel_m2 = np.zeros(n_channels, dtype=np.float64)  # sum of squared deviations from current mean
    
    print("\nRunning Welford online algorithm (memory-efficient)...")
    success_count = 0
    fail_count = 0
    
    for idx in tqdm(sample_indices, desc="Processing samples"):
        sample = samples[idx]
        timestamp = sample['timestamp']
        
        # Use file index for fast lookup
        if timestamp not in file_index:
            fail_count += 1
            continue
        
        sample_success = True
        
        # Online update per channel
        for ch_idx in range(n_channels):
            if ch_idx not in file_index[timestamp]:
                sample_success = False
                break
            
            try:
                fpath = file_index[timestamp][ch_idx]
                data = np.load(fpath)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                data_flat = data.flatten()
                
                # Welford's online update
                for value in data_flat:
                    channel_count[ch_idx] += 1
                    delta = value - channel_mean[ch_idx]
                    channel_mean[ch_idx] += delta / channel_count[ch_idx]
                    delta2 = value - channel_mean[ch_idx]
                    channel_m2[ch_idx] += delta * delta2
                    
            except Exception as e:
                sample_success = False
                break
        
        if sample_success:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nProcessing complete: {success_count} succeeded, {fail_count} failed")
    
    # Compute final mean and std from accumulated Welford statistics
    means = []
    stds = []
    
    print("\nPer-channel statistics:")
    print(f"{'Ch':^6} {'Variable':^15} {'Mean':^15} {'Std':^15} {'N':^15}")
    print("-" * 80)
    
    for i, channel_name in enumerate(channel_names):
        if channel_count[i] == 0:
            print(f"Warning: channel {i} ({channel_name}) has no data!")
            means.append(0.0)
            stds.append(1.0)
            continue
        
        mean_val = channel_mean[i]
        
        # std = sqrt(M2 / N)
        if channel_count[i] > 1:
            std_val = np.sqrt(channel_m2[i] / channel_count[i])
        else:
            std_val = 1.0
        
        if std_val == 0:
            std_val = 1.0
        
        means.append(float(mean_val))
        stds.append(float(std_val))
        
        # Print first 5 and last 4 channels
        if i < 5 or i >= n_channels - 4:
            print(f"{i:^6d} {channel_name:^15s} {mean_val:>15.4f} {std_val:>15.4f} {channel_count[i]:>15d}")
        elif i == 5:
            print(f"{'...':^6} {'...':^15} {'...':^15} {'...':^15} {'...':^15}")
    
    mean_tensor = torch.tensor(means, dtype=torch.float32)
    std_tensor = torch.tensor(stds, dtype=torch.float32)
    
    stats = {
        'mean': mean_tensor,
        'std': std_tensor,
        'channel_names': channel_names,
        'n_channels': n_channels,
        'n_samples_used': success_count,
        'computed_from': 'filtered_train_set',
        'filter_threshold': Config.DataProcessing.SAMPLE_FILTER_THRESHOLD,
        'algorithm': 'welford_online'
    }
    
    # Save stats
    os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)
    with open(stats_save_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"\nNormalization stats saved: {stats_save_path}")
    print(f"Based on {success_count} filtered training samples (Welford online algorithm)")
    
    return stats


# ============================================================================
# Dataset
# ============================================================================

class DirectLoadDataset(Dataset):
    """Loads ERA5 and GPM IMERG data directly from per-variable .npy files."""
    
    def __init__(self, metadata, file_indices=None, transform=None, 
                 latlon_generator=None, precipitation_threshold=None, file_index=None):
        
        if precipitation_threshold is None:
            precipitation_threshold = Config.DataProcessing.PRECIPITATION_THRESHOLD
        
        self.metadata = metadata
        self.transform = transform
        self.latlon_generator = latlon_generator
        self.precipitation_threshold = precipitation_threshold
        
        self.samples = metadata['samples']
        if file_indices is not None:
            self.samples = [self.samples[i] for i in file_indices]
        
        self.precip_dir = metadata['precip_dir']
        self.era5_base = metadata['era5_base']
        self.file_index = file_index
        
        # Build variable directory mapping (upper-air then surface)
        self.var_dirs = []
        for var in Config.ERA5Variables.UPPER_VARS:
            for level in Config.ERA5Variables.PRESSURE_LEVELS:
                dir_name = f"{var}{level}hpa"
                self.var_dirs.append(os.path.join(self.era5_base, dir_name))
        for var in Config.ERA5Variables.SURFACE_VARS:
            self.var_dirs.append(os.path.join(self.era5_base, var))
        
        if self.latlon_generator is not None:
            self.latlon_grids = self.latlon_generator.get_grids()
        else:
            self.latlon_grids = None
        
        print(f"DirectLoadDataset initialized: {len(self.samples)} samples, threshold={precipitation_threshold}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        timestamp = sample['timestamp']
        
        # 1. Load precipitation and transpose to match ERA5 orientation
        precip_path = os.path.join(self.precip_dir, sample['precip_file'])
        precipitation = np.load(precip_path)
        # GPM IMERG requires transpose + vertical flip to align with ERA5 grid
        precipitation = precipitation.T
        precipitation = np.flip(precipitation, axis=0) 
        
        # 2. Load 82 ERA5 channels (no transpose needed)
        era5_list = []
        
        if self.file_index and timestamp in self.file_index:
            for var_idx in range(len(self.var_dirs)):
                if var_idx in self.file_index[timestamp]:
                    fpath = self.file_index[timestamp][var_idx]
                    try:
                        data = np.load(fpath)
                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                        era5_list.append(data)
                    except:
                        era5_list.append(np.zeros(precipitation.shape, dtype=np.float32))
                else:
                    era5_list.append(np.zeros(precipitation.shape, dtype=np.float32))
        else:
            for var_dir in self.var_dirs:
                files = [f for f in os.listdir(var_dir) if timestamp in f and f.endswith('.npy')]
                
                if not files:
                    era5_list.append(np.zeros(precipitation.shape, dtype=np.float32))
                else:
                    fpath = os.path.join(var_dir, files[0])
                    data = np.load(fpath)
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    era5_list.append(data)
        
        era5_array = np.stack(era5_list, axis=0)
        era5_tensor = torch.from_numpy(era5_array).float()
        
        # 3. Normalize ERA5 inputs
        if self.transform:
            era5_tensor = self.transform(era5_tensor)
        
        # 4. Append lat/lon coordinate channels
        if self.latlon_grids is not None:
            if self.latlon_grids.shape[1:] != era5_tensor.shape[1:]:
                latlon_resized = F.interpolate(
                    self.latlon_grids.unsqueeze(0),
                    size=era5_tensor.shape[1:],
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)
                input_data = torch.cat([era5_tensor, latlon_resized], dim=0)
            else:
                input_data = torch.cat([era5_tensor, self.latlon_grids], dim=0)
        else:
            input_data = era5_tensor
        
        # 5. Generate binary target mask
        precipitation_mask = (precipitation > self.precipitation_threshold).astype(np.float32)
        target = torch.from_numpy(precipitation_mask).float().unsqueeze(0)
        
        return input_data, target


def compute_class_weights_simple(train_dataset, sample_size=None, mask_generator=None):
    """
    Estimates the positive class weight for BCE loss from a random subset of the training set.

    Args:
        train_dataset: training dataset
        sample_size: number of samples to use for estimation
        mask_generator: optional core region mask (statistics computed within core domain only)
    """
    if sample_size is None:
        sample_size = Config.Loss.CLASS_WEIGHT_SAMPLE_SIZE
    
    print("Computing class weights...")
    
    pos_count = 0
    neg_count = 0
    
    sample_size = min(len(train_dataset), sample_size)
    sample_indices = np.random.choice(len(train_dataset), sample_size, replace=False)
    
    # Get core region mask
    mask = None
    if mask_generator is not None:
        mask = mask_generator.get_mask()
    
    for idx in tqdm(sample_indices, desc="Estimating class distribution"):
        _, target = train_dataset[idx]
        target_np = target.numpy()
        
        # Apply spatial mask
        if mask is not None:
            mask_np = mask.numpy()
            if mask_np.shape != target_np.shape:
                mask_np = np.broadcast_to(mask_np, target_np.shape)
            target_masked = target_np[mask_np > 0.5]
        else:
            target_masked = target_np
        
        pos_count += np.sum(target_masked > 0.5)
        neg_count += np.sum(target_masked <= 0.5)
    
    pos_ratio = pos_count / (pos_count + neg_count)
    neg_ratio = neg_count / (pos_count + neg_count)
    
    pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0
    
    print(f"Positive ratio: {pos_ratio:.4f}")
    print(f"Negative ratio: {neg_ratio:.4f}")
    print(f"pos_weight: {pos_weight:.4f}")
    
    return pos_weight


# ============================================================================
# Loss Functions (with spatial mask support)
# ============================================================================

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: model output probabilities
            targets: binary ground truth
            mask: spatial region mask, shape=(H, W), values 0 or 1
        """
        eps = 1e-7
        inputs = torch.clamp(inputs, eps, 1 - eps)
        bce_loss = -(targets * torch.log(inputs) * self.pos_weight + 
                     (1 - targets) * torch.log(1 - inputs))
        
        # Apply spatial mask
        if mask is not None:
            # Expand mask to (B, 1, H, W)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            # Broadcast and apply
            mask = mask.expand_as(bce_loss)
            bce_loss = bce_loss * mask
            
            # Average over masked region only
            return bce_loss.sum() / (mask.sum() + eps)
        else:
            return bce_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth or Config.Loss.DICE_SMOOTH
    
    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: model output probabilities
            targets: binary ground truth
            mask: spatial region mask
        """
        # Apply spatial mask before computing Dice
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            mask = mask.expand_as(inputs)
            inputs_masked = inputs * mask
            targets_masked = targets * mask
        else:
            inputs_masked = inputs
            targets_masked = targets
        
        inputs_flat = inputs_masked.view(-1)
        targets_flat = targets_masked.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=None, dice_weight=None, pos_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight or Config.Loss.BCE_WEIGHT
        self.dice_weight = dice_weight or Config.Loss.DICE_WEIGHT
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: model output probabilities
            targets: binary ground truth
            mask: spatial region mask
        """
        bce = self.bce_loss(inputs, targets, mask)
        dice = self.dice_loss(inputs, targets, mask)
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================================
# Model Architecture
# ============================================================================

class UNetBinary(nn.Module):
    """UNet binary classifier with 84 input channels."""
    
    def __init__(self, in_channels=None, base_channels=None, dropout_rate=None):
        super().__init__()
        
        if in_channels is None:
            in_channels = Config.Model.IN_CHANNELS
        if base_channels is None:
            base_channels = Config.Model.BASE_CHANNELS
        if dropout_rate is None:
            dropout_rate = Config.Model.DROPOUT_RATE
        
        self.dropout_rate = dropout_rate
        c = base_channels
        c_mults = Config.Model.CHANNEL_MULTS
        
        self.init_conv = nn.Conv2d(in_channels, c, 3, padding=1)
        self.init_norm = nn.BatchNorm2d(c)
        self.init_act = nn.ReLU(inplace=True)
        self.init_dropout = nn.Dropout2d(p=dropout_rate * 0.5)
        
        dims = [c * mult for mult in c_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        self.encoders.append(self._make_layer(c, c, dropout_rate * 0.5))
        
        for i, (d_in, d_out) in enumerate(in_out):
            self.pools.append(nn.MaxPool2d(2))
            layer_dropout = dropout_rate * min(1.0, 0.5 + i * 0.2)
            self.encoders.append(self._make_layer(d_in, d_out, layer_dropout))
        
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            self.upsamples.append(nn.ConvTranspose2d(d_out, d_in, kernel_size=2, stride=2))
            layer_dropout = dropout_rate * 0.3
            self.decoders.append(self._make_layer(d_in * 2, d_in, layer_dropout))
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.2),
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
    
    def _make_layer(self, in_ch, out_ch, dropout_rate=0.3):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5)
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        orig_size = x.shape[2:]
        
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = self.init_act(x)
        x = self.init_dropout(x)
        initial = x.clone()
        
        skip_connections = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skip_connections.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        
        skip_connections = skip_connections[:-1]
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        if x.shape[2:] != initial.shape[2:]:
            x = F.interpolate(x, size=initial.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, initial], dim=1)
        x = self.final_conv(x)
        
        if x.shape[2:] != orig_size:
            x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=True)
        
        return x


# ============================================================================
# Training Function
# ============================================================================

def train_stage1_csi(model, train_loader, val_loader, optimizer, scheduler, 
                     criterion, mask_generator=None, num_epochs=None, device=None, 
                     use_dropout_schedule=None, threshold_value=None):
    """
    Trains a single binary classifier using CSI as the primary evaluation metric.

    Args:
        mask_generator: CoreRegionMaskGenerator instance for restricting loss to core domain
        threshold_value: precipitation threshold this model is trained for (used for naming checkpoints)
    """
    
    if num_epochs is None:
        num_epochs = Config.Training.NUM_EPOCHS
    if device is None:
        device = Config.Training.DEVICE
    if use_dropout_schedule is None:
        use_dropout_schedule = Config.Model.USE_DROPOUT_SCHEDULE
    
    model = model.to(device)
    
    # Get core region mask
    core_mask = None
    if mask_generator is not None:
        core_mask = mask_generator.get_mask(device)
        print(f"\nCore region loss enabled: {mask_generator.core_ratio*100:.2f}% of grid")
    
    dropout_scheduler = None
    if use_dropout_schedule:
        dropout_scheduler = DropoutScheduler(model)
    
    best_loss = float('inf')
    best_csi = 0.0  # CSI used as primary model selection criterion
    best_threshold = 0.5
    
    model_dir = Config.DataPaths.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate model filename from precipitation threshold
    if threshold_value is not None:
        threshold_str = str(threshold_value).replace('.', 'p')
        best_model_name = f'best_model_csi_thresh{threshold_str}.pth'
        history_name = f'stage1_history_csi_thresh{threshold_str}.json'
    else:
        best_model_name = Config.DataPaths.BEST_MODEL_NAME
        history_name = Config.DataPaths.HISTORY_NAME
    
    best_model_path = os.path.join(model_dir, best_model_name)
    
    patience = Config.Training.EARLY_STOPPING_PATIENCE
    early_stopping_counter = 0
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'train_csi': [],
        'val_loss': [],
        'val_acc': [],
        'val_csi': [],
        'learning_rates': [],
        'best_thresholds': [],
        'dropout_rates': [],
        'core_region_enabled': mask_generator is not None,
        'metric_type': 'CSI',
        'precipitation_threshold': threshold_value
    }
    
    print(f"\n{'='*80}")
    print(f"Training started - CSI metric, precipitation threshold={threshold_value}")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        if use_dropout_schedule:
            current_dropout = dropout_scheduler.step(epoch)
            training_history['dropout_rates'].append(current_dropout)
        
        # ==================== Training ====================
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_csi = 0.0  # CSI accumulator
        running_threshold = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute loss restricted to core domain
            loss = criterion(outputs, targets, mask=core_mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.Training.GRAD_CLIP_NORM)
            optimizer.step()
            
            running_loss += loss.item()
            
            with torch.no_grad():
                # Find optimal CSI threshold within core domain
                batch_threshold, batch_csi = find_optimal_threshold_csi(outputs, targets, mask=core_mask)
                running_threshold += batch_threshold
                running_csi += batch_csi
                
                # Accuracy within core domain
                predicted = (outputs > batch_threshold).float()
                if core_mask is not None:
                    mask_expanded = core_mask.unsqueeze(0).unsqueeze(0).expand_as(predicted)
                    pred_masked = predicted[mask_expanded > 0.5]
                    targ_masked = targets[mask_expanded > 0.5]
                    acc = (pred_masked == targ_masked).float().mean().item()
                else:
                    acc = (predicted == targets).float().mean().item()
                running_acc += acc
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = running_acc / len(train_loader)
        epoch_train_csi = running_csi / len(train_loader)
        
        # ==================== Validation ====================
        model.eval()
        val_loss = 0.0
        
        all_val_outputs = []
        all_val_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                # Compute validation loss restricted to core domain
                loss = criterion(outputs, targets, mask=core_mask)
                val_loss += loss.item()
                
                all_val_outputs.append(outputs)
                all_val_targets.append(targets)
        
        all_val_outputs = torch.cat(all_val_outputs, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)
        
        # Find optimal CSI threshold on full validation set within core domain
        epoch_val_threshold, epoch_val_csi = find_optimal_threshold_csi(
            all_val_outputs, all_val_targets, mask=core_mask
        )
        
        # Validation accuracy within core domain
        predicted = (all_val_outputs > epoch_val_threshold).float()
        if core_mask is not None:
            mask_expanded = core_mask.unsqueeze(0).unsqueeze(0).expand_as(predicted)
            pred_masked = predicted[mask_expanded > 0.5]
            targ_masked = all_val_targets[mask_expanded > 0.5]
            epoch_val_acc = (pred_masked == targ_masked).float().mean().item()
        else:
            epoch_val_acc = (predicted == all_val_targets).float().mean().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
        
        # ==================== Log History ====================
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc)
        training_history['train_csi'].append(epoch_train_csi)
        training_history['val_loss'].append(epoch_val_loss)
        training_history['val_acc'].append(epoch_val_acc)
        training_history['val_csi'].append(epoch_val_csi)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        training_history['best_thresholds'].append(float(epoch_val_threshold))
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, CSI: {epoch_train_csi:.4f}')
        print(f'  Val   - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, CSI: {epoch_val_csi:.4f}')
        print(f'  Best Threshold: {epoch_val_threshold:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        if mask_generator:
            print(f'  (Core region {mask_generator.core_ratio*100:.1f}% of grid)')
        
        # Save best model by CSI
        if epoch_val_csi > best_csi:
            best_csi = epoch_val_csi
            best_loss = epoch_val_loss
            best_threshold = epoch_val_threshold
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'best_val_csi': best_csi,
                'best_val_loss': best_loss,
                'best_threshold': best_threshold,
                'training_history': training_history,
                'core_region_config': {
                    'enabled': mask_generator is not None,
                    'lon_range': Config.DataProcessing.LOSS_LON_RANGE if mask_generator else None,
                    'lat_range': Config.DataProcessing.LOSS_LAT_RANGE if mask_generator else None,
                },
                'metric_type': 'CSI',
                'precipitation_threshold': threshold_value
            }, best_model_path)
            print(f"  Best model saved! CSI: {best_csi:.4f}, Threshold: {best_threshold:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"  No CSI improvement ({early_stopping_counter}/{patience})")
        
        if early_stopping_counter >= patience:
            print(f"\nEarly stopping triggered (no CSI improvement for {patience} epochs)")
            break
        
        print("-" * 80)
    
    # Save training history
    history_path = os.path.join(model_dir, history_name)
    with open(history_path, 'w') as f:
        history_serializable = {k: [float(x) if not isinstance(x, (bool, type(None), str)) else x 
                                     for x in v] if isinstance(v, list) else v 
                                for k, v in training_history.items()}
        json.dump(history_serializable, f, indent=2)
    
    print(f'\n{"="*80}')
    print(f'Training complete! (precipitation threshold={threshold_value})')
    print(f'{"="*80}')
    print(f'Best val CSI: {best_csi:.4f}')
    print(f'Best threshold: {best_threshold:.4f}')
    print(f'Best model: {best_model_path}')
    print(f'Training history: {history_path}')
    
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training pipeline: trains 15 binary classifiers across precipitation thresholds."""
    
    print("\n" + "=" * 80)
    print(" " * 15 + "Stage1 Multi-threshold Training - CSI Version")
    print("=" * 80)
    
    print("\nKey design choices:")
    print("  - CSI (Critical Success Index) used as evaluation metric")
    print("  - 15 independent binary classifiers across precipitation thresholds")
    print("  - Threshold range: 0.1-100 mm/6h")
    print("  - Loss computed within core domain (110-130E, 20-40N)")
    
    # Step 1: Build metadata
    print("\n" + "=" * 80)
    print("Step 1: Build metadata (shared across all models)")
    print("=" * 80)
    metadata = build_metadata_from_existing_files()
    
    # Step 2: Build ERA5 file index
    print("\n" + "=" * 80)
    print("Step 2: Build ERA5 file index (shared across all models)")
    print("=" * 80)
    
    var_dirs = []
    for var in Config.ERA5Variables.UPPER_VARS:
        for level in Config.ERA5Variables.PRESSURE_LEVELS:
            var_dirs.append(os.path.join(metadata['era5_base'], f"{var}{level}hpa"))
    for var in Config.ERA5Variables.SURFACE_VARS:
        var_dirs.append(os.path.join(metadata['era5_base'], var))
    
    file_index = FileIndexBuilder.build_file_index(
        era5_base=metadata['era5_base'],
        var_dirs=var_dirs,
        cache_path=Config.DataPaths.FILE_INDEX_CACHE_PATH,
        force_rebuild=Config.DataProcessing.FORCE_REBUILD_FILE_INDEX
    )
    
    # Step 3: Precipitation-based sample filtering (shared threshold)
    print("\n" + "=" * 80)
    print("Step 3: Precipitation sample filtering (shared across all models)")
    print("=" * 80)
    valid_indices, precipitation_stats = filter_samples_by_precipitation(metadata)
    
    # Step 4: Train/validation split (shared)
    print("\n" + "=" * 80)
    print("Step 4: Train/validation split (shared across all models)")
    print("=" * 80)
    
    train_indices = []
    val_indices = []
    
    total_samples = len(metadata['samples'])
    for idx in valid_indices:
        if idx >= total_samples:
            continue
        
        sample = metadata['samples'][idx]
        year = sample['year']
        if year in Config.DataSplit.TRAIN_YEARS:
            train_indices.append(idx)
        elif year in Config.DataSplit.VAL_YEARS:
            val_indices.append(idx)
    
    print(f"Train: {len(train_indices)} samples")
    print(f"Val:   {len(val_indices)} samples")
    
    # Step 5: Compute normalization statistics (shared)
    print("\n" + "=" * 80)
    print("Step 5: Compute normalization stats from filtered training set (shared)")
    print("=" * 80)
    
    normalization_stats = compute_normalization_stats_from_filtered(
        metadata=metadata,
        train_indices=train_indices,
        file_index=file_index
    )
    
    mean_all = normalization_stats['mean']
    std_all = normalization_stats['std']
    input_transform = CustomInputNormalize(mean_all, std_all)
    
    # Step 6: Create lat/lon generator and core region mask (shared)
    print("\n" + "=" * 80)
    print("Step 6: Create lat/lon generator and core region mask (shared)")
    print("=" * 80)
    
    latlon_generator = LatLonGridGenerator(grid_shape=metadata['grid_shape'])
    mask_generator = CoreRegionMaskGenerator(grid_shape=metadata['grid_shape'])
    
    # Step 7: Train one model per precipitation threshold
    print("\n" + "=" * 80)
    print("Step 7: Multi-threshold training loop")
    print("=" * 80)
    
    thresholds = Config.DataProcessing.TRAINING_THRESHOLDS
    print(f"Training {len(thresholds)} models for thresholds: {thresholds}")
    
    for threshold_idx, current_threshold in enumerate(thresholds):
        print("\n" + "=" * 80)
        print(f"Model {threshold_idx + 1}/{len(thresholds)}: precipitation threshold = {current_threshold} mm/6h")
        print("=" * 80)
        
        # Build datasets for current threshold
        train_dataset = DirectLoadDataset(
            metadata=metadata,
            file_indices=train_indices,
            transform=input_transform,
            latlon_generator=latlon_generator,
            precipitation_threshold=current_threshold,
            file_index=file_index
        )
        
        val_dataset = DirectLoadDataset(
            metadata=metadata,
            file_indices=val_indices,
            transform=input_transform,
            latlon_generator=latlon_generator,
            precipitation_threshold=current_threshold,
            file_index=file_index
        )
        
        # Estimate class weights for current threshold
        pos_weight = compute_class_weights_simple(train_dataset, mask_generator=mask_generator)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.Training.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.Training.NUM_WORKERS,
            pin_memory=Config.Training.PIN_MEMORY
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.Training.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.Training.NUM_WORKERS,
            pin_memory=Config.Training.PIN_MEMORY
        )
        
        # Initialize fresh model for each threshold
        model = UNetBinary()
        criterion = CombinedLoss(pos_weight=pos_weight)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.Training.INITIAL_LR,
            weight_decay=Config.Training.WEIGHT_DECAY
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=Config.Training.LR_SCHEDULER_FACTOR,
            patience=Config.Training.LR_SCHEDULER_PATIENCE,
            verbose=True,
            min_lr=Config.Training.LR_SCHEDULER_MIN_LR
        )
        
        # Train
        model = train_stage1_csi(
            model, train_loader, val_loader, optimizer, scheduler, criterion,
            mask_generator=mask_generator,
            threshold_value=current_threshold
        )
        
        # Free memory before next threshold
        del model, optimizer, scheduler, criterion, train_loader, val_loader
        del train_dataset, val_dataset
        torch.cuda.empty_cache()
        
        print(f"\nModel {threshold_idx + 1}/{len(thresholds)} complete!\n")
    
    print("\n" + "=" * 80)
    print("All models trained successfully!")
    print("=" * 80)
    print(f"Total models: {len(thresholds)}")
    print(f"Model directory: {Config.DataPaths.MODEL_DIR}")


if __name__ == '__main__':
    main()