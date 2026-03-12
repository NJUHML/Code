# CBC-Diffusion: A Two-Stage Framework for Extreme Precipitation Forecasting

This repository contains the full implementation of a two-stage deep learning framework for extreme precipitation forecasting over eastern China (110–130°E, 20–40°N). The framework combines **Cascade Binary Classification (CBC)** with **diffusion model post-processing** to achieve high-resolution, probabilistic forecasts of extreme events from ERA5 reanalysis inputs.

The code accompanies the paper submitted to *JGR: Machine Learning and Computation*.

---

## Framework Overview

**Stage 1 — Cascade Binary Classification (CBC)**
Reformulates precipitation prediction as 15 independent binary classification tasks across thresholds τᵢ ∈ {0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0} mm 6h⁻¹. Each threshold model is an independent U-Net, eliminating inter-threshold optimization conflicts. Outputs are integrated via **first-negative hierarchical stacking** to produce a 0.25° diagnostic field.

**Stage 2 — Diffusion Post-Processing**
A DDPM-based UNet performs **joint bias calibration and spatial downscaling** from 0.25° to 0.1°. Following SDEdit (Meng et al., 2022), the CBC field is upsampled and partially corrupted to t*=100 (out of T=1000), then denoised to generate 10 ensemble members (Mean / Median / P80 / P90).

---

## File Structure

```
.
├── stage1_cbc_train.py           # Stage 1: Train 15 independent binary U-Net classifiers
├── stage1_cbc_inference.py       # Stage 1: Run inference with all 15 trained models
├── stage1_cbc_ensemble.py        # Stage 1: First-negative hierarchical stacking ensemble
├── stage2_diffusion_train.py     # Stage 2: Train diffusion UNet on GPM IMERG
├── stage2_diffusion_inference.py # Stage 2: SDEdit post-processing on CBC output
└── README.md
```

---

## Data Format

This section describes the exact file structure expected by each script.

### ERA5 Input Data

The ERA5 data must be preprocessed into per-variable `.npy` files and organized as follows:

```
ERA5_BASE_PATH/
├── u50hpa/
│   ├── u50hpa_2013010100.npy
│   ├── u50hpa_2013010106.npy
│   └── ...
├── u100hpa/
├── ...
├── u1000hpa/
├── v50hpa/
├── ...
├── t50hpa/
├── ...
├── z50hpa/
├── ...
├── rh50hpa/          # relative humidity
├── ...
├── vertical_velocity50hpa/
├── ...
├── u10/              # surface: 10-m zonal wind
├── v10/              # surface: 10-m meridional wind
├── t2m/              # surface: 2-m temperature
└── msl/              # surface: mean sea level pressure
```

**Directory naming convention:** `{variable}{level}hpa` for upper-air (e.g., `u925hpa`, `t500hpa`), variable name only for surface (e.g., `u10`, `t2m`).

**File naming convention:** `{variable}{level}hpa_{timestamp}.npy` where timestamp is a 10-digit string `YYYYMMDDHH` (e.g., `u925hpa_2013010106.npy`). The timestamp must be detectable via the regex pattern `_?(\d{10})\.npy$`.

**Array shape:** Each `.npy` file contains a 2D array of shape `(97, 97)` covering the extended domain 108–132°E, 18–42°N at 0.25° resolution.

**Total channels:** 6 upper-air variables × 13 pressure levels (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa) + 4 surface variables = 82 channels. Two lat/lon coordinate channels are appended automatically by the dataloader, yielding 84 channels total.

**Normalization:** ERA5 channels are normalized using per-channel Z-score statistics computed from the training set. Statistics are saved to `normalization_stats_filteredmore.pkl` on first run and reused thereafter.

---

### GPM IMERG Precipitation (for CBC training)

```
PRECIP_6H_DIR/
├── precipitation_2013010100.npy
├── precipitation_2013010106.npy
└── ...
```

**File naming:** `*_{timestamp}.npy` (10-digit `YYYYMMDDHH`).

**Array shape:** Each file contains a 2D array. Note that GPM IMERG is stored in (lon, lat) order — the dataloader automatically applies `.T` (transpose) followed by `np.flip(..., axis=0)` (vertical flip) to convert to (lat, lon) with north-up orientation matching ERA5.

**Content:** 6-hour accumulated precipitation in mm, coarsened to 0.25° to match ERA5 grid.

**Sample filtering:** Only samples with domain-averaged precipitation ≥ 1.0 mm are retained for training (`Config.DataProcessing.SAMPLE_FILTER_THRESHOLD = 1.0`). All test samples are kept regardless of this threshold.

---

### GPM IMERG Raw Files (for diffusion training)

```
GPM_BASE_DIR/
├── R2013/
│   ├── 3B-HHR.MS.MRG.3IMERG.20130101-S000000-E002959.0000.V07B.HDF5.SUB.nc4
│   ├── 3B-HHR.MS.MRG.3IMERG.20130101-S003000-E005959.0030.V07B.HDF5.SUB.nc4
│   └── ...
├── R2014/
└── ...
```

**Format:** Standard GPM IMERG V07B Final Run half-hourly `.nc4` files. The dataloader pairs consecutive 30-minute files by hour and accumulates them to 6-hour totals. Files with non-standard naming or incomplete hour pairs are skipped automatically.

**Array shape after loading:** `(200, 200)` covering 110–130°E, 20–40°N at native 0.1° resolution.

---

### Precomputed Normalization Statistics

`normalization_stats_filteredmore.pkl` — generated automatically by `stage1_cbc_train.py` on first run. Must be present when running `stage1_cbc_inference.py`. Contains per-channel mean and std for all 82 ERA5 meteorological channels.

---

## Pipeline Execution

```bash
# Stage 1
python stage1_cbc_train.py       # trains 15 models, saves checkpoints + normalization stats
python stage1_cbc_inference.py   # runs inference on 2023 test set
python stage1_cbc_ensemble.py    # applies first-negative stacking, saves CBC precipitation fields

# Stage 2
python stage2_diffusion_train.py   # trains diffusion UNet on GPM IMERG
python stage2_diffusion_inference.py  # SDEdit post-processing on CBC output
```

Before running, update the path variables at the top of each script (`ERA5_BASE_PATH`, `PRECIP_6H_DIR`, `MODEL_DIR`, etc.) to match your local data layout.

---

## Data Split

| Split      | Years                           |
|------------|---------------------------------|
| Training   | 2013–2015, 2017–2019, 2021–2022 |
| Validation | 2016, 2020                      |
| Test       | 2023                            |

---

## Model Architecture

### Stage 1: CBC U-Net (×15 models)
- **Input:** (84, 97, 97) — 84-channel ERA5 over extended domain
- **Encoder:** 4 downsampling stages, channel dims 64→128→256→512, dual conv + BN + ReLU + Dropout
- **Output:** Sigmoid probability map (1, 97, 97); loss computed on core 81×81 grid only
- **Parameters:** ~7–10M per model; 117.87M total across 15 models
- **Training:** Combined BCE + Dice loss (α=0.8); optimal p_th by CSI grid search on validation set

### Stage 2: Diffusion UNet
- **Input/output:** (1, 200, 200) at 0.1°
- **Channels:** 128 base, multipliers (1, 2, 4, 8), sinusoidal time embeddings
- **Noise schedule:** Cosine (Nichol & Dhariwal, 2021), T=1000; inference at t*=100
- **Parameters:** 135.88M; ~37 s per 10-member ensemble

---

## Computational Environment

All experiments conducted on a single **NVIDIA L40 GPU**. Each CBC model trains in ~30–40 minutes (all 15 can run in parallel). End-to-end inference: ~38 s per 6-hourly sample.

---

## Dependencies

```bash
pip install torch numpy scipy einops netCDF4 tqdm matplotlib cartopy
```

---

## Citation

> [To be updated upon publication]

---

## References

- Ho et al. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
- Meng et al. (2022). SDEdit: Guided image synthesis and editing with stochastic differential equations. *ICLR*.
- Nichol & Dhariwal (2021). Improved denoising diffusion probabilistic models. *ICML*.
- Ronneberger et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.