# CM4nowcast

## 1. Introduction

CM4nowcast provides the implementation of a Customized Multi-Scale (CM) deep learning framework for storm/precipitation nowcasting.
The framework is designed to better capture complex multi-scale spatiotemporal variability and improve forecast quality, especially for high-impact storms.

If you use this code in academic work, please cite the paper listed in Section 4.

Repository path:
`Code/CM4nowcast/` (this folder)

## 2. Required Python Packages

Recommended environment:
- Python >= 3.8
- CUDA + PyTorch (if using GPU)

Typical dependencies:
- numpy
- scipy
- pandas
- xarray
- matplotlib
- tqdm
- pyyaml
- torch
- torchvision (optional)
- tensorboard (optional)

Example installation:

pip install -r requirements.txt

## 3. Usage Instructions

A typical workflow is:
1) prepare data
2) train the model
3) run inference / nowcast
4) evaluate results

### 3.1 Data preparation

Place your data under:

- data/raw/
- data/processed/

### 3.2 Training

Example:

python train.py --config configs/train.yaml

### 3.3 Inference / Nowcasting

Example:

python inference.py --config configs/infer.yaml --ckpt checkpoints/best.pt

### 3.4 Evaluation

Example:

python evaluate.py --config configs/eval.yaml

## 4. Citation

If you use this code in your research, please cite:

Yang, S., & Yuan, H. (2023). A Customized Multi-Scale Deep Learning Framework for Storm Nowcasting.
Geophysical Research Letters, 50, e2023GL103979.
https://doi.org/10.1029/2023GL103979

BibTeX:

@article{{yang2023cm4nowcast,
  title   = {{A Customized Multi-Scale Deep Learning Framework for Storm Nowcasting}},
  author  = {{Yang, Shangshang and Yuan, Huiling}},
  journal = {{Geophysical Research Letters}},
  volume  = {{50}},
  pages   = {{e2023GL103979}},
  year    = {{2023}},
  doi     = {{10.1029/2023GL103979}}
}}

## 5. License

Add a license file (e.g., MIT / Apache-2.0) if distributing publicly.
