
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re



# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Test configuration class."""
    
    class DataPaths:
        ERA5_BASE_PATH = '/data/daily/Diffusion1106/era5_processed_6h_multilayer_2023/'
        PRECIP_6H_DIR = '/data/daily/Diffusion1106/precipitation_6h2023/'
        METADATA_PATH = '/data/daily/paper1123/zhenduantest/direct_load_metaall.pkl'
        FILTER_CACHE_PATH = '/data/daily/paper1123/zhenduantest/precipitation_filter_cache_6h.pkalll'
        STATS_SAVE_PATH = '/data/daily/paper1123/finalzhenduan/normalization_stats_filteredmore.pkl'
        MODEL_DIR = '/data/daily/paper1123/finalzhenduan/finalzhenduanmodel1/'
        
        # List of model checkpoints to evaluate
        MODEL_FILES = [
            'best_model_csi_thresh0p1.pth',
            'best_model_csi_thresh0p5.pth',
            'best_model_csi_thresh1p0.pth',
            'best_model_csi_thresh2p5.pth',
            'best_model_csi_thresh5p0.pth',
            'best_model_csi_thresh10p0.pth',
            'best_model_csi_thresh20p0.pth',
            'best_model_csi_thresh30p0.pth',
            'best_model_csi_thresh40p0.pth',
            'best_model_csi_thresh50p0.pth',
            'best_model_csi_thresh60p0.pth',
            'best_model_csi_thresh70p0.pth',
            'best_model_csi_thresh80p0.pth',
            'best_model_csi_thresh90p0.pth',
            'best_model_csi_thresh100p0.pth',
        ]
        
        # Output directory for test results
        TEST_RESULTS_BASE_DIR = '/data/daily/paper1123/zhenduantestallfinal/'
        TEST_VISUALIZATIONS_DIR = 'visualizations/'
        PREDICTIONS_OUTPUT_DIR = 'predictions/'
    
    class ERA5Variables:
        """ERA5 variable configuration (6 upper-air vars x 13 levels + 4 surface vars)."""
        UPPER_VARS = ['u', 'v', 't', 'z', 'rh', 'vertical_velocity']
        PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        SURFACE_VARS = ['u10', 'v10', 't2m', 'msl']
        N_UPPER_CHANNELS = 6 * 13  # 78
        N_SURFACE_CHANNELS = 4
        N_LATLON_CHANNELS = 2
        N_TOTAL_CHANNELS = 78 + 4 + 2  # 84 total channels
    
    class DataProcessing:
        LON_RANGE = (108, 132)  # Full input domain
        LAT_RANGE = (18, 42)    # Full input domain
        
        # Spatial crop for saved prediction outputs
        OUTPUT_LON_RANGE = (110, 130)
        OUTPUT_LAT_RANGE = (20, 40)
        
        SAMPLE_FILTER_THRESHOLD = 0.001  # Consistent with training
    
    class DataSplit:
        VAL_YEARS = [2023]
    
    class Model:
        IN_CHANNELS = 84  # 82 ERA5 + 2 lat/lon
        GUIYIHUA = 82     # channels requiring normalization
        BASE_CHANNELS = 64
        CHANNEL_MULTS = (1, 2, 4, 8)
        DROPOUT_RATE = 0.05
    
    class Testing:
        BATCH_SIZE = 12
        NUM_WORKERS = 6
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        PIN_MEMORY = True
        
        NUM_VISUALIZATION_SAMPLES = 20
        SAVE_VISUALIZATIONS = True
        
        SAVE_PROBABILITY = True   # whether to save raw probability maps
        SAVE_BINARY = True        # whether to save thresholded binary maps



def extract_threshold_from_filename(filename):
    """
    Parses the precipitation threshold from a model checkpoint filename.
    Examples:
      best_model_csi_thresh0p1.pth   -> 0.1
      best_model_csi_thresh10p0.pth  -> 10.0
      best_model_csi_thresh100p0.pth -> 100.0
    """
    import re
    
    # Match the pattern 'thresh<int>p<decimal>'
    match = re.search(r'thresh(\d+)p(\d+)', filename)
    if match:
        integer_part = match.group(1)
        decimal_part = match.group(2)
        return float(f"{integer_part}.{decimal_part}")
    
    return None


# ============================================================================
# Crop and Interpolation Utilities
# ============================================================================

def crop_predictions(predictions, original_lon_range, original_lat_range, 
                     target_lon_range, target_lat_range):

    # Ensure 4D tensor
    if predictions.dim() == 3:
        predictions = predictions.unsqueeze(1)
        squeeze_back = True
    else:
        squeeze_back = False
    
    B, C, H, W = predictions.shape
    
    # Compute pixel indices from fractional positions
    lon_start_ratio = (target_lon_range[0] - original_lon_range[0]) / (original_lon_range[1] - original_lon_range[0])
    lon_end_ratio = (target_lon_range[1] - original_lon_range[0]) / (original_lon_range[1] - original_lon_range[0])
    lat_start_ratio = (target_lat_range[0] - original_lat_range[0]) / (original_lat_range[1] - original_lat_range[0])
    lat_end_ratio = (target_lat_range[1] - original_lat_range[0]) / (original_lat_range[1] - original_lat_range[0])
    
    lon_start_idx = int(lon_start_ratio * W)
    lon_end_idx = int(lon_end_ratio * W)
    lat_start_idx = int(lat_start_ratio * H)
    lat_end_idx = int(lat_end_ratio * H)
    
    # Crop without interpolation
    cropped = predictions[:, :, lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx]
    
    if squeeze_back:
        cropped = cropped.squeeze(1)
    
    return cropped

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
    """Per-channel normalization for ERA5 input."""
    
    def __init__(self, mean, std):
        self.mean = mean.unsqueeze(-1).unsqueeze(-1)
        self.std = std.unsqueeze(-1).unsqueeze(-1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / (self.std + 1e-8)



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


def build_metadata_from_existing_files(precip_dir, era5_base, metadata_path, force_recompute=False):
    """Scans precipitation and ERA5 directories to build a sample list keyed by matched timestamps."""
    
    # Load from cache
    if os.path.exists(metadata_path) and not force_recompute:
        print(f"Loading metadata from cache: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"[OK] Loaded {len(metadata['samples'])} samples")
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
    
    print(f"[OK] Matched timestamps: {len(common_times)}")
    
    if len(common_times) == 0:
        print("\n[ERROR] No matching timestamps found between precipitation and ERA5 data")
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
        'n_channels': Config.Model.GUIYIHUA,
        'time_range': {
            'start': common_times[0].isoformat(),
            'end': common_times[-1].isoformat()
        }
    }
    
    # Save metadata
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"[OK] Metadata saved: {metadata_path}")
    print(f"[OK] Total samples: {len(samples)}, grid shape: {grid_shape}")
    
    return metadata


def save_prediction_results(predictions, timestamps, threshold, output_dir):
    """
    Saves per-sample prediction outputs (probability maps and binary masks) to .npy files.
    Predictions are spatially cropped to OUTPUT_LON/LAT_RANGE before saving.

    Args:
        predictions: raw model outputs [N, 1, H, W]
        timestamps: list of timestamp strings
        threshold: decision threshold for binarization
        output_dir: base directory for saved outputs
    """
    
    print(f"\n=== Saving prediction outputs ===")
    print(f"Output dir: {output_dir}")
    print(f"Crop region: lon {Config.DataProcessing.OUTPUT_LON_RANGE}, lat {Config.DataProcessing.OUTPUT_LAT_RANGE}")
    
    # Create output subdirectories
    prob_dir = os.path.join(output_dir, 'probability')
    binary_dir = os.path.join(output_dir, 'binary')
    
    if Config.Testing.SAVE_PROBABILITY:
        os.makedirs(prob_dir, exist_ok=True)
    if Config.Testing.SAVE_BINARY:
        os.makedirs(binary_dir, exist_ok=True)
    
    # Crop predictions to output domain
    print(f"Shape before crop: {predictions.shape}")
    predictions_cropped = crop_predictions(
        predictions,
        original_lon_range=Config.DataProcessing.LON_RANGE,
        original_lat_range=Config.DataProcessing.LAT_RANGE,
        target_lon_range=Config.DataProcessing.OUTPUT_LON_RANGE,
        target_lat_range=Config.DataProcessing.OUTPUT_LAT_RANGE
    )
    print(f"Shape after crop: {predictions_cropped.shape}")
    
    pred_np = predictions_cropped.cpu().numpy()
    pred_binary = (pred_np > threshold).astype(np.float32)
    
    n_samples = len(timestamps)
    
    for i in tqdm(range(n_samples), desc="Saving predictions"):
        timestamp = timestamps[i]
        
        # Drop channel dim: [1, H, W] -> [H, W]
        prob_array = pred_np[i, 0, :, :]
        binary_array = pred_binary[i, 0, :, :]
        
        if Config.Testing.SAVE_PROBABILITY:
            prob_file = os.path.join(prob_dir, f'pred_prob_{timestamp}.npy')
            np.save(prob_file, prob_array)
        
        if Config.Testing.SAVE_BINARY:
            binary_file = os.path.join(binary_dir, f'pred_binary_{timestamp}.npy')
            np.save(binary_file, binary_array)
    
    # Save metadata sidecar
    meta_info = {
        'n_samples': n_samples,
        'threshold': float(threshold),
        'grid_shape': list(pred_np.shape[2:]),
        'lon_range': Config.DataProcessing.OUTPUT_LON_RANGE,
        'lat_range': Config.DataProcessing.OUTPUT_LAT_RANGE,
        'original_lon_range': Config.DataProcessing.LON_RANGE,
        'original_lat_range': Config.DataProcessing.LAT_RANGE,
        'timestamps': timestamps,
        'saved_probability': Config.Testing.SAVE_PROBABILITY,
        'saved_binary': Config.Testing.SAVE_BINARY,
        'save_time': datetime.now().isoformat()
    }
    
    meta_file = os.path.join(output_dir, 'prediction_metadata.json')
    with open(meta_file, 'w') as f:
        json.dump(meta_info, f, indent=2)
    
    print(f"[OK] Saved {n_samples} samples")
    if Config.Testing.SAVE_PROBABILITY:
        print(f"  Probability maps: {prob_dir}")
    if Config.Testing.SAVE_BINARY:
        print(f"  Binary masks:     {binary_dir}")
    print(f"  Metadata:         {meta_file}")


def visualize_predictions(inputs, predictions, targets, threshold, 
                         save_dir, sample_indices):
    """Saves side-by-side visualizations of predicted probability and binary prediction."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    pred_np = predictions.cpu().numpy()
    pred_binary = (pred_np > threshold).astype(np.float32)
    
    num_samples = min(len(sample_indices), Config.Testing.NUM_VISUALIZATION_SAMPLES)
    
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        im0 = axes[0].imshow(pred_np[i, 0], cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title('Prediction Probability', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        axes[1].imshow(pred_binary[i, 0], cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title(f'Binary Prediction (threshold={threshold:.3f})', fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{sample_indices[i]:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] Saved {num_samples} visualizations to: {save_dir}")


# ============================================================================
# Dataset
# ============================================================================

class DirectLoadDataset(Dataset):
    """Loads ERA5 and GPM IMERG data directly from per-variable .npy files (includes vertical_velocity)."""
    
    def __init__(self, metadata, file_indices=None, transform=None, 
                 latlon_generator=None, precipitation_threshold=None):
        
        # precipitation_threshold must match the threshold used during training
        
        self.metadata = metadata
        self.transform = transform
        self.latlon_generator = latlon_generator
        self.precipitation_threshold = precipitation_threshold
        
        self.samples = metadata['samples']
        if file_indices is not None:
            self.samples = [self.samples[i] for i in file_indices]
        
        self.precip_dir = metadata['precip_dir']
        self.era5_base = metadata['era5_base']
        
        # Build variable directory mapping (upper-air then surface)
        self.var_dirs = []
        for var in Config.ERA5Variables.UPPER_VARS:      # 6 upper-air variables
            for level in Config.ERA5Variables.PRESSURE_LEVELS:  # 13 pressure levels
                dir_name = f"{var}{level}hpa"
                self.var_dirs.append(os.path.join(self.era5_base, dir_name))
        for var in Config.ERA5Variables.SURFACE_VARS:    # 4 surface variables
            self.var_dirs.append(os.path.join(self.era5_base, var))
        
        # Verify channel count matches model config
        assert len(self.var_dirs) == Config.Model.GUIYIHUA, \
            f"Channel count mismatch: {len(self.var_dirs)} != {Config.Model.GUIYIHUA}"
        
        if self.latlon_generator is not None:
            self.latlon_grids = self.latlon_generator.get_grids()
        else:
            self.latlon_grids = None
        
        print(f"[OK] DirectLoadDataset initialized: {len(self.samples)} samples, {len(self.var_dirs)} ERA5 channels")
    
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
        # 2. Load ERA5 channels (82 upper-air + surface, no transpose needed)
        era5_list = []
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
        
        # 4. Append lat/lon coordinate channels (82 + 2 = 84 total)
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
        
        return input_data, target, timestamp  # also return timestamp for file naming


# ============================================================================
# Model Architecture
# ============================================================================

class UNetBinary(nn.Module):
    """UNet binary classifier with 84 input channels (82 ERA5 + 2 lat/lon)."""
    
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
# Test Functions
# ============================================================================

def test_model(model, test_loader, threshold, device=None):
    """Runs inference on the test set and returns predictions and timestamps."""
    
    if device is None:
        device = Config.Testing.DEVICE
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_timestamps = []
    
    print("\n=== Running model inference ===")
    print(f"Decision threshold: {threshold:.4f}")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc='Testing'):
            inputs, targets, timestamps = batch_data
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            
            all_predictions.append(outputs.cpu())
            all_timestamps.extend(timestamps)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    
    return all_predictions, all_timestamps


# ============================================================================
# Single-Model Test Driver
# ============================================================================

def test_single_model(model_file, test_indices, device, metadata, 
                      input_transform, latlon_generator):
    """Loads a single checkpoint, runs inference, and saves prediction outputs."""
    
    # Parse precipitation threshold from filename
    threshold_value = extract_threshold_from_filename(model_file)
    if threshold_value is None:
        print(f"\n[WARNING] Cannot extract threshold from filename: {model_file}")
        return False
    
    print("\n" + "=" * 80)
    print(f"Testing model: {model_file}")
    print(f"Precipitation threshold: {threshold_value} mm/6h")
    print("=" * 80)
    
    # Create per-threshold results directory
    threshold_str = str(threshold_value).replace('.', '_')
    save_dir = os.path.join(Config.DataPaths.TEST_RESULTS_BASE_DIR, f'threshold_{threshold_str}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load checkpoint
    model_path = os.path.join(Config.DataPaths.MODEL_DIR, model_file)
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file not found: {model_path}")
        return False
    
    checkpoint = torch.load(model_path, map_location=device)
    best_threshold = checkpoint.get('best_threshold', 0.5)
    epoch = checkpoint.get('epoch', 0)
    
    print(f"[OK] Checkpoint loaded (epoch {epoch + 1}, decision threshold {best_threshold:.4f})")
    
    # Instantiate model and load weights
    model = UNetBinary()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset
    test_dataset = DirectLoadDataset(
        metadata=metadata,
        file_indices=test_indices,
        transform=input_transform,
        latlon_generator=latlon_generator,
        precipitation_threshold=threshold_value
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.Testing.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.Testing.NUM_WORKERS,
        pin_memory=Config.Testing.PIN_MEMORY
    )
    
    print(f"[OK] Test set: {len(test_dataset)} samples")
    
    # Run inference
    predictions, timestamps = test_model(model, test_loader, best_threshold, device)
    
    # Save prediction outputs
    pred_output_dir = os.path.join(save_dir, Config.DataPaths.PREDICTIONS_OUTPUT_DIR)
    save_prediction_results(predictions, timestamps, best_threshold, pred_output_dir)
    
    # Save visualizations
    if Config.Testing.SAVE_VISUALIZATIONS:
        vis_dir = os.path.join(save_dir, Config.DataPaths.TEST_VISUALIZATIONS_DIR)
        num_vis = min(Config.Testing.NUM_VISUALIZATION_SAMPLES, len(predictions))
        vis_indices = np.random.choice(len(predictions), num_vis, replace=False)
        visualize_predictions(
            None, predictions[vis_indices], None,
            best_threshold, vis_dir, vis_indices
        )
    
    print(f"\n[OK] Model {model_file} done, outputs saved to: {save_dir}")
    return True


def main():
    """Main test pipeline: evaluates all 15 binary classifiers on the test set."""
    
    print("=" * 80)
    print(" " * 5 + "Stage1 Batch Model Evaluation (with vertical velocity + eval region + prediction output)")
    print("=" * 80)
    
    device = Config.Testing.DEVICE
    print(f"\nDevice: {device}")
    print(f"[OK] Model input channels: {Config.Model.IN_CHANNELS} (82 ERA5 + 2 lat/lon)")
    print(f"[OK] ERA5 variables: {', '.join(Config.ERA5Variables.UPPER_VARS)}")
    print(f"\n[OK] Models to evaluate: {len(Config.DataPaths.MODEL_FILES)}")
    
    # Step 1: Load or build metadata
    print("\n" + "=" * 80)
    print("Step 1: Load or build test set metadata")
    print("=" * 80)
    
    metadata_path = Config.DataPaths.METADATA_PATH
    force_rebuild = False
    
    if os.path.exists(metadata_path) and not force_rebuild:
        print(f"Loading metadata from cache: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        if metadata.get('precip_dir') != Config.DataPaths.PRECIP_6H_DIR or \
           metadata.get('era5_base') != Config.DataPaths.ERA5_BASE_PATH:
            print("[WARNING] Data path mismatch in cached metadata, rebuilding...")
            force_rebuild = True
    else:
        force_rebuild = True
    
    if force_rebuild:
        print("Building test set metadata...")
        metadata = build_metadata_from_existing_files(
            precip_dir=Config.DataPaths.PRECIP_6H_DIR,
            era5_base=Config.DataPaths.ERA5_BASE_PATH,
            metadata_path=metadata_path,
            force_recompute=True
        )
    
    print(f"[OK] Total samples: {metadata['n_samples']}")
    print(f"[OK] Grid shape:    {metadata['grid_shape']}")
    print(f"[OK] Time range:    {metadata['time_range']['start']} ~ {metadata['time_range']['end']}")
    
    # Step 2: Filter samples by precipitation
    print("\n" + "=" * 80)
    print("Step 2: Precipitation-based sample filtering")
    print("=" * 80)
    
    precipitation_threshold = Config.DataProcessing.SAMPLE_FILTER_THRESHOLD
    filter_cache_path = Config.DataPaths.FILTER_CACHE_PATH
    force_recompute = False
    
    if os.path.exists(filter_cache_path) and not force_recompute:
        print(f"Loading filter cache: {filter_cache_path}")
        with open(filter_cache_path, 'rb') as f:
            filter_data = pickle.load(f)
        valid_indices = filter_data['valid_indices']
        print(f"[OK] Valid samples: {len(valid_indices)}")
    else:
        print(f"Filtering samples with threshold: {precipitation_threshold} mm/6h")
        
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
            'max_precip': np.max(precipitation_values),
            'min_precip': np.min(precipitation_values),
            'mean_precip': np.mean(precipitation_values),
            'std_precip': np.std(precipitation_values),
            'threshold': precipitation_threshold,
            'valid_count': len(valid_indices),
            'total_count': total_samples,
            'filter_ratio': len(valid_indices) / total_samples if total_samples > 0 else 0.0
        }
        
        print(f"\n=== Filter results ===")
        print(f"Total: {total_samples}  |  Valid: {len(valid_indices)}  |  Ratio: {len(valid_indices)/total_samples*100:.2f}%")
        
        cache_data = {
            'valid_indices': valid_indices,
            'precipitation_stats': precipitation_stats,
            'total_samples': total_samples,
            'threshold': precipitation_threshold
        }
        os.makedirs(os.path.dirname(filter_cache_path), exist_ok=True)
        with open(filter_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[OK] Filter cache saved: {filter_cache_path}")
    
    # Step 3: Select test samples
    print("\n" + "=" * 80)
    print("Step 3: Prepare test dataset from filtered samples")
    print("=" * 80)
    
    test_indices = []
    
    if Config.DataSplit.VAL_YEARS is None:
        test_indices = valid_indices
        print(f"[OK] Using all filtered samples as test set")
    else:
        for idx in valid_indices:
            sample = metadata['samples'][idx]
            if sample['year'] in Config.DataSplit.VAL_YEARS:
                test_indices.append(idx)
        print(f"  Test years: {Config.DataSplit.VAL_YEARS}")
    
    print(f"[OK] Filtered valid samples: {len(valid_indices)}")
    print(f"[OK] Test set samples: {len(test_indices)}")
    
    if len(test_indices) == 0:
        print("\n[ERROR] No test samples found for the specified years")
        return
    
    # Step 4: Load normalization statistics
    print("\n" + "=" * 80)
    print("Step 4: Load normalization statistics")
    print("=" * 80)
    
    with open(Config.DataPaths.STATS_SAVE_PATH, 'rb') as f:
        stats = pickle.load(f)
    
    input_transform = CustomInputNormalize(stats['mean'], stats['std'])
    print(f"[OK] Normalization stats loaded: {len(stats['mean'])} channels")
    
    # Step 5: Create lat/lon generator
    print("\n" + "=" * 80)
    print("Step 5: Create lat/lon grid generator")
    print("=" * 80)
    
    latlon_generator = LatLonGridGenerator(grid_shape=metadata['grid_shape'])
    print(f"[OK] Lat/lon grid generator created")
    
    # Step 6: Batch inference of all models
    print("\n" + "=" * 80)
    print("Step 6: Batch inference of all models")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for i, model_file in enumerate(Config.DataPaths.MODEL_FILES, 1):
        print(f"\n{'='*80}")
        print(f"Progress: [{i}/{len(Config.DataPaths.MODEL_FILES)}]")
        print(f"{'='*80}")
        
        try:
            ok = test_single_model(
                model_file=model_file,
                test_indices=test_indices,
                device=device,
                metadata=metadata,
                input_transform=input_transform,
                latlon_generator=latlon_generator
            )
            if ok:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to test {model_file}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print("\n" + "=" * 80)
    print("Batch inference complete!")
    print("=" * 80)
    print(f"\nSummary: {successful} succeeded, {failed} failed (of {len(Config.DataPaths.MODEL_FILES)})")
    print(f"[OK] Prediction outputs saved to: {Config.DataPaths.TEST_RESULTS_BASE_DIR}")


if __name__ == '__main__':
    main()