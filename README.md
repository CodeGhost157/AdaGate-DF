# Low Resource and Low Resolution Deepfake Detection Benchmark

This repository contains cleaned, script-based code for the deepfake detection experiments originally developed in the notebooks under `notebooks/original/`. The project focuses on binary real versus fake face-frame classification under low-resource and low-resolution settings. It compares a proposed lightweight gated dual-branch detector against neural and classical baselines.

## What is being compared

| Model | Source notebook | Main idea | Dataset usage |
| --- | --- | --- | --- |
| Gated Dual-Branch Deepfake Detector | `our_methodology.ipynb` | Lightweight RGB plus frequency model with early exits and a quality-aware route gate | FaceForensics++ C23 |
| MaD-CoRN | `MaD_CoRN_Pipeline.ipynb` | Multi-scale attention and convolutional reservoir network | FaceForensics++ C23 and Celeb-DF style image splits |
| ShuffleNetV2 | `ShuffleNet.ipynb` | Lightweight CNN baseline using ImageNet pretraining | FaceForensics++ C23 and Celeb-DF |
| EfficientNet-B4 | `DiffusionFake.ipynb` | Larger pretrained timm baseline | Celeb-DF v2 image dataset |
| DeFakeHop++ Approximation | `DeFakeHop++.ipynb` | PCA and LightGBM based classical feature pipeline | FaceForensics++ C23 and Celeb-DF |

The experiments evaluate accuracy, AUC, precision, recall, F1, uncertainty rate, training time, inference speed, and performance across source resolutions of 128, 224, 256, and 384 pixels.

## Repository layout

```text
.
├── configs/                         # Editable experiment configs
├── docs/                            # Notes and paper-facing documentation
├── legacy_exports/                  # Notebook code exported to .py for traceability
├── notebooks/original/              # Uploaded original notebooks, unchanged
├── scripts/                         # Command-line entrypoints
├── src/deepfake_lowres/             # Reusable package code
│   ├── data/                        # Manifest loading, preprocessing, datasets, transforms
│   ├── models/                      # Architectures and model wrappers
│   ├── results/                     # Results extracted from current notebooks
│   └── training/                    # Training and evaluation loops
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

For conda:

```bash
conda env create -f environment.yml
conda activate deepfake-lowres
pip install -e .
```

## Data preparation

### FaceForensics++ C23

The original preprocessing notebook extracts frames and creates a `metadata.csv` with these columns:

```text
frame_path, frame_name, video_path, video_id, split, label, label_name, frame_index_in_video, sample_index
```

Run the cleaned preprocessing script:

```bash
python scripts/preprocess_ffpp.py \
  --raw-root /path/to/FaceForensics++_C23 \
  --output-root data/ffpp_preprocessed \
  --target-fps 2 \
  --max-frames-per-video 8 \
  --jpeg-quality 95 \
  --num-workers 4
```

After this, update the `metadata_csv` field in the relevant config files:

```yaml
metadata_csv: data/ffpp_preprocessed/metadata.csv
```

### Celeb-DF v2 image dataset

The expected layout is:

```text
Celeb_V2/
├── Train/
│   ├── real/
│   └── fake/
├── Val/
│   ├── real/
│   └── fake/
└── Test/
    ├── real/
    └── fake/
```

Create optional CSV manifests:

```bash
python scripts/build_celebdf_manifest.py \
  --root /path/to/Celeb_V2 \
  --out-dir data/celebdf_manifests
```

For training, update `celebdf_root` in `configs/efficientnet_b4_celebdf.yaml`.

## Running experiments

### Proposed gated dual-branch model

```bash
python scripts/train_gated_dual_branch.py \
  --config configs/gated_dual_branch_ffpp.yaml
```

Outputs are written to `outputs/gated_dual_branch_ffpp/`.

### ShuffleNetV2 baseline

```bash
python scripts/train_torch_model.py \
  --config configs/shufflenet_ffpp_oversampled.yaml
```

### MaD-CoRN baseline

```bash
python scripts/train_torch_model.py \
  --config configs/madcorn_ffpp.yaml
```

### EfficientNet-B4 baseline

```bash
python scripts/train_torch_model.py \
  --config configs/efficientnet_b4_celebdf.yaml
```

### DeFakeHop++ approximate classical baseline

```bash
python scripts/train_defakehoppp.py \
  --config configs/defakehoppp_ffpp_oversampled.yaml
```

This one can take a while because it fits PCA models, builds feature matrices, and then trains LightGBM.

## Current notebook results

These are the results extracted from the executed outputs in the current uploaded notebooks. Use them as a starting point, then regenerate all tables after final reruns in the cleaned repository.

| Model | Dataset | Accuracy | AUC | F1 | Training time |
| --- | --- | ---: | ---: | ---: | ---: |
| EfficientNet-B4 Real/Fake | Celeb-DF v2 image dataset | 0.9968 | 0.9999 | 0.9968 | 70.11 min |
| ShuffleNetV2 FF++ real oversampled | FaceForensics++ C23 | 0.7822 | 0.5383 | 0.8757 | 502.42 min |
| MaD-CoRN-FFPP | FaceForensics++ C23 | 0.8571 | 0.5592 | 0.9230 | 646.90 min |
| Gated Dual-Branch Deepfake Detector | FaceForensics++ C23 | 0.8571 | 0.5163 | 0.9230 | 334.93 min |

### Resolution results from current notebooks

EfficientNet-B4 on Celeb-DF:

| Resolution | Accuracy | AUC |
| ---: | ---: | ---: |
| 128 | 0.8326 | 0.9467 |
| 224 | 0.9968 | 0.9999 |
| 256 | 0.9969 | 0.9999 |
| 384 | 0.9661 | 0.9972 |

ShuffleNetV2 on FaceForensics++ C23:

| Resolution | Accuracy | AUC |
| ---: | ---: | ---: |
| 128 | 0.6282 | 0.5415 |
| 224 | 0.8515 | 0.5336 |
| 256 | 0.8088 | 0.5334 |
| 384 | 0.8405 | 0.5355 |

MaD-CoRN on FaceForensics++ C23:

| Resolution | Accuracy | AUC |
| ---: | ---: | ---: |
| 128 | 0.8571 | 0.5639 |
| 224 | 0.8571 | 0.5569 |
| 256 | 0.8571 | 0.5590 |
| 384 | 0.8571 | 0.5575 |

Gated Dual-Branch on FaceForensics++ C23:

| Resolution | Accuracy | AUC | Fast route | Medium route | Full route |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.8571 | 0.5043 | 0.00% | 100.00% | 0.00% |
| 224 | 0.8571 | 0.5253 | 88.64% | 11.36% | 0.00% |
| 256 | 0.8571 | 0.5214 | 89.78% | 10.22% | 0.00% |
| 384 | 0.8571 | 0.5202 | 89.37% | 10.29% | 0.34% |

The DeFakeHop++ notebook output included preprocessing and feature-fitting progress, but the uploaded execution output did not include a completed final metric table. The cleaned training script will save `test_results.csv`, `accuracy_based_on_resolution.csv`, and `training_time.csv` after completion.

## Outputs saved by each run

Each experiment folder is designed to contain:

```text
training_period_results.csv
accuracy_based_on_resolution.csv
inference_time.csv
test_results.csv
training_time.csv
experiment_summary.csv or summary.json
best model checkpoint
```

## Reproducibility notes

The original notebooks used hard-coded Kaggle cache paths like `/home/jovyan/.cache/kagglehub/...`. Those paths were moved into YAML configs so the code can run on other machines. Random seeds are set through `seed_everything`, but exact GPU runs may still differ slightly because PyTorch and CUDA kernels can be nondeterministic.

## Paper-ready checklist

Before publishing the GitHub repository, rerun every experiment from the scripts and replace the current notebook-extracted results with fresh script-generated CSVs. Add your final paper citation, dataset access instructions, environment details, GPU model, and any licenses required by the datasets.

