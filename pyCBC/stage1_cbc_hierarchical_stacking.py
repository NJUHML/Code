
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class Config:

    class Paths:
        RESULTS_BASE_DIR = '/data/daily/paper1123/zhenduantestallfinal/'
        TRUE_PRECIP_DIR = '/data/daily/Diffusion1106/precipitation_6h2023/'
        ENSEMBLE_OUTPUT_DIR = '/data/daily/paper1123/zhenduantestallfinal/ensemble_resultsright/'
        ENSEMBLE_PRECIP_DIR = 'ensemble_precipitationright/'
    
    class Thresholds:
        """Precipitation thresholds (must match training)."""
        VALUES = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 
                  40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        
        @staticmethod
        def get_folder_name(threshold):
            return f"threshold_{str(threshold).replace('.', '_')}"
    
    class Grid:
        SHAPE = (81, 81)
        LON_RANGE = (110, 130)
        LAT_RANGE = (20, 40)



# ============================================================================
# Core Ensemble Functions - Hierarchical Stacking (First-Negative Logic)
# ============================================================================

def load_binary_predictions(threshold_folders, timestamp):
    """Loads binary prediction arrays for all thresholds at a given timestamp."""
    predictions = {}
    
    for threshold in Config.Thresholds.VALUES:
        folder_name = Config.Thresholds.get_folder_name(threshold)
        folder_path = os.path.join(Config.Paths.RESULTS_BASE_DIR, folder_name)
        
        binary_file = os.path.join(folder_path, 'predictions', 'binary', 
                                   f'pred_binary_{timestamp}.npy')
        
        if os.path.exists(binary_file):
            predictions[threshold] = np.load(binary_file)
        else:
            print(f"  [WARNING] Missing file: {binary_file}")
            predictions[threshold] = None
    
    return predictions


def ensemble_precipitation_v2(binary_predictions):
    """
    Combines binary classifier outputs into a precipitation field using
    first-negative hierarchical stacking logic.

    Traverses thresholds from low to high. At each grid point, the first
    negative prediction halts the traversal; the preceding (last positive)
    threshold value is assigned as the precipitation estimate. If all
    classifiers are positive, the highest threshold is used. If all are
    negative, precipitation is set to zero.

    Args:
        binary_predictions: dict {threshold: binary_array [H, W]}

    Returns:
        precipitation: estimated precipitation field [H, W]
    """
    # Sort thresholds ascending
    thresholds = sorted(binary_predictions.keys())
    
    # Determine grid shape from the first available prediction
    for threshold, pred in binary_predictions.items():
        if pred is not None:
            shape = pred.shape
            break
    else:
        raise ValueError("All predictions are missing!")
    
    precipitation = np.zeros(shape, dtype=np.float32)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            last_positive_threshold = 0.0
            
            for threshold in thresholds:
                pred = binary_predictions[threshold]
                if pred is None:
                    continue
                
                if pred[i, j] > 0.5:   # positive: continue to next threshold
                    last_positive_threshold = threshold
                else:                   # negative: stop traversal
                    break
            
            precipitation[i, j] = last_positive_threshold
    
    return precipitation


def load_true_precipitation(timestamp):
    """Loads and preprocesses the GPM IMERG ground-truth precipitation for a given timestamp."""
    precip_files = [f for f in os.listdir(Config.Paths.TRUE_PRECIP_DIR) 
                    if timestamp in f and f.endswith('.npy')]
    
    if not precip_files:
        return None
    
    precip_path = os.path.join(Config.Paths.TRUE_PRECIP_DIR, precip_files[0])
    precip = np.load(precip_path)
    
    # Transpose and flip to align with ERA5 grid orientation
    precip = precip.T
    precip = np.flip(precip, axis=0)
    
    # Crop to output domain [81, 81]
    precip_cropped = precip[8:89, 8:89]
    
    return precip_cropped



# ============================================================================
# Main
# ============================================================================

def get_all_timestamps():
    """Returns a sorted list of all available timestamps from the first threshold's binary output directory."""
    first_folder = Config.Thresholds.get_folder_name(Config.Thresholds.VALUES[0])
    binary_dir = os.path.join(Config.Paths.RESULTS_BASE_DIR, first_folder, 
                              'predictions', 'binary')
    
    if not os.path.exists(binary_dir):
        raise FileNotFoundError(f"Prediction directory not found: {binary_dir}")
    
    files = [f for f in os.listdir(binary_dir) if f.startswith('pred_binary_')]
    timestamps = [f.replace('pred_binary_', '').replace('.npy', '') for f in files]
    
    return sorted(timestamps)


def main():
    """Main pipeline: hierarchical stacking ensemble across all timestamps."""
    
    print("=" * 80)
    print(" " * 15 + "Multi-model Precipitation Ensemble - Hierarchical Stacking")
    print(" " * 22 + "(First-Negative Logic)")
    print("=" * 80)
    
    print("\nEnsemble logic:")
    print("  Traverse thresholds from low to high. At each grid point, stop at")
    print("  the first negative prediction and assign the preceding threshold as")
    print("  the precipitation estimate.")
    print("\n  Example: 10+, 40+, 70-, 80+ -> result = 40 mm")
    print("           (stops at 70, does not check 80)")
    
    # Create output directories
    os.makedirs(Config.Paths.ENSEMBLE_OUTPUT_DIR, exist_ok=True)
    ensemble_precip_dir = os.path.join(Config.Paths.ENSEMBLE_OUTPUT_DIR, 
                                       Config.Paths.ENSEMBLE_PRECIP_DIR)
    os.makedirs(ensemble_precip_dir, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Thresholds ({len(Config.Thresholds.VALUES)}): {Config.Thresholds.VALUES}")
    print(f"  Grid shape: {Config.Grid.SHAPE}")
    print(f"  Output dir: {Config.Paths.ENSEMBLE_OUTPUT_DIR}")
    
    # Step 1: Collect all available timestamps
    print("\n" + "=" * 80)
    print("Step 1: Collect available timestamps")
    print("=" * 80)
    
    timestamps = get_all_timestamps()
    print(f"[OK] Found {len(timestamps)} timestamps")
    print(f"  Time range: {timestamps[0]} ~ {timestamps[-1]}")
    
    # Step 2: Build ensemble precipitation fields
    print("\n" + "=" * 80)
    print("Step 2: Build ensemble precipitation fields (first-negative logic)")
    print("=" * 80)
    
    ensemble_precips = []
    true_precips = []
    valid_timestamps = []
    
    for timestamp in tqdm(timestamps, desc="Ensemble"):
        try:
            # Load binary predictions from all 15 classifiers
            binary_preds = load_binary_predictions(Config.Paths.RESULTS_BASE_DIR, timestamp)
            
            # Skip samples with too many missing predictions
            valid_preds = sum(1 for p in binary_preds.values() if p is not None)
            if valid_preds < len(Config.Thresholds.VALUES) * 0.5:
                print(f"  [WARNING] Too many missing predictions for {timestamp}, skipping")
                continue
            
            # Apply first-negative hierarchical stacking
            ensemble_precip = ensemble_precipitation_v2(binary_preds)
            
            # Load ground-truth precipitation
            true_precip = load_true_precipitation(timestamp)
            if true_precip is None:
                print(f"  [WARNING] Missing ground truth for {timestamp}, skipping")
                continue
            
            # Save ensemble output
            output_file = os.path.join(ensemble_precip_dir, f'ensemble_v2_{timestamp}.npy')
            np.save(output_file, ensemble_precip)
            
            ensemble_precips.append(ensemble_precip)
            true_precips.append(true_precip)
            valid_timestamps.append(timestamp)
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {timestamp}: {e}")
            continue
    
    print(f"\n[OK] Successfully processed {len(ensemble_precips)} samples")
    print(f"[OK] Ensemble fields saved to: {ensemble_precip_dir}")
    
    if len(ensemble_precips) == 0:
        print("\n[ERROR] No valid ensemble fields produced, exiting")
        return
    
    print("\n" + "=" * 80)
    print("Ensemble complete!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  Ensemble fields: {ensemble_precip_dir}")
    print(f"  Samples:         {len(ensemble_precips)}")


if __name__ == '__main__':
    main()