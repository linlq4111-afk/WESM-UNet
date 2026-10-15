# WESM-UNet: Wavelet-Enhanced Edge-Aware Semantic Mamba U-Net

PyTorch implementation of WESM-UNet for medical image segmentation (Synapse / ACDC / chest CT).

## Installation

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU (`mamba-ssm` needs CUDA kernels; CPU-only execution is not supported).

## Data layout

Place each dataset under `./data` (paths are configurable in `configs/`):

```
./data/
  synapse/        # Synapse slices + lists (train_vol/image lists)
  ACDC/database/  # ACDC .h5 database
  chest_ct/       # in-house chest CT slices (not released)
```

- Synapse: 18 training / 12 test volumes (2211 training slices), 8 organs.
- ACDC: 100 training / 50 test patients (RV / Myo / LV).
- Chest CT: in-house cohort (n = 236); data are not publicly released.

## Training

```bash
# Synapse (300 epochs, batch 12, lr 4e-4, AdamW, AMP, single RTX 4090)
python train_synapse.py

# ACDC (300 epochs, batch 12, lr 4e-4)
python train_acdc.py

# Chest CT (200 epochs, batch 12, lr 1e-4)
python train_hypertension_synapse.py
```

Loss: `0.4 * CrossEntropy + 0.6 * Dice`. Augmentation: random rotation / flip / Gaussian noise.
Checkpoints are written next to the training log; select by the held-out epoch-wise metric.

## Evaluation

Use the same `test_single_volume` utilities in `utils.py` via the engine scripts
(`engine_synapse.py`, `engine_acdc.py`, `engine_hypertension.py`) to compute
mean DSC and HD95 (Synapse / chest CT) or mean Dice (ACDC).

## Model

- `models/vmunet/wesmunet.py`: WESM-UNet (fine-grained stem + WGD + EADB + SASF)
- `models/vmunet/wesm_backbone.py`: Mamba backbone blocks
- `models/vmunet/wavelet.py`: Haar DWT utilities for WGD
- `models/vmunet/stem.py`: fine-grained embedding stem block
- `models/baselines/`: U-Net / VM-UNet baselines used for comparison

Pretrained ImageNet weights for the backbone can be loaded via `load_ckpt_path`
(see `WESMUNet.load_from()`); downsampling (WGD) layers are trained from scratch.

## Notes

- All inputs are resized to 224x224 by default.
- Results in the paper: Synapse mDSC 85.77 / HD95 15.05; ACDC mean Dice 92.20;
  chest CT mDSC 86.60 / HD95 4.79.
