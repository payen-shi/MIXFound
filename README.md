# MIXFound

This repository provides a complete pipeline for **fundus image classification** based on several powerful visual backbones (VisionFM, RETFound, FLAIR, CLIP) and **MIXFound**.  

It includes:

- **Feature extraction** scripts for each backbone.
- **Linear classification decoders** for downstream disease classification.
- **Mixture of Foundation Models** code (MIXFound) with ROC and calibration analysis.
- Reproducible **training & evaluation** pipelines with AUC/F1/confusion matrix.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Pretrained Weights](#pretrained-weights)
- [Usage](#usage)
  - [1. Feature Extraction](#1-feature-extraction)
  - [2. Train Classification Decoders](#2-train-classification-decoders)
  - [3. Multi‑Model Fusion (MIXFound)](#3-multi-model-fusion-mixfound)
- [Evaluation & Outputs](#evaluation--outputs)
- [Customization](#customization)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Multiple backbones**
  - VisionFM (fundus foundation model)
  - RETFound
  - FLAIR
  - OpenAI CLIP (ViT-L/14)

- **Flexible classification decoders**
  - Linear heads (`ClsHead`) on frozen features.
  - Support for multi-class classification.

- **Rich evaluation metrics**
  - Accuracy, Precision, Recall, F1
  - Class-wise and macro **ROC‑AUC**
  - **Bootstrap 95% CI** for AUC
  - Confusion matrices + per‑class sensitivity / specificity

- **Multi‑model fusion (MIXFound)**
  - AUC‑based weighting across multiple backbones.
  - ROC comparison plots across models and fusion.

- **Reproducible & scalable training**
  - Configurable via `argparse` (dataset, Task A–G, seeds, batch size, etc.).
  - Distributed Data Parallel (DDP) support.

---

## Repository Structure

Github/
├── environment.yml                # Conda environment specification
├── MIXFound.py                    # Multi-model fusion (MIXFound) main script
├── utils.py                       # Shared utilities (DDP, metrics, logging, etc.)
├── dataset/                       # Dataset utilities / wrappers
├── Classification/
│   ├── CLIP_based_classifier.py   # CLIP-based classifier (ViT-L/14 features)
│   ├── FLAIR_based_classifier.py  # FLAIR-based classifier
│   ├── RETFound_based_classifier.py # RETFound-based classifier
│   ├── VisionFM_based_classifier.py  # VisionFM-based classifier
│   ├── utils.py                   # Classification-specific utilities
│   ├── evaluation_funcs.py        # Evaluation helpers
│   └── ...                        # Other related utilities
└── Feature_Extraction/
    ├── RETFound/
    │   ├── RETFound_Feature_Extractor.py
    │   └── models_vit.py
    ├── FLAIR/
    │   └── FLAIR_Feature_Extractor.py
    └── ...                        # (optional) other backbones> **Note**: The repository is designed to be used from the `Github/` directory as the project root.

---

## Environment Setup

We recommend using **conda**:

# 1. Create environment from provided file
conda env create -f environment.yml -n visionfm

# 2. Activate environment
conda activate visionfm

# 3. (Optional) Export current environment to requirements.txt
pip freeze > requirements.txt If you prefer a pure `pip` workflow, you can inspect `environment.yml` and install the listed packages manually.

---

## Data Preparation

By default, the classification scripts expect a fundus dataset organized as:

dataset/
└── fundus/
    ├── A/.../G/                   # training, evaluation, and test images
    │   └── Age related macular degeneration (AMD)    
    │   └── Glaucoma     
    │   └── High myopia     
    │   └── Normal fundus     
└── training/
    │   └── training_labels.txt    # training labels (path;label)
└── evaluation/
    │   └── evaluation_labels.txt    # evaluation labels (path;label)
└── training/
    ├── test_labels             # validation / test images

--data_path /path/to/dataset/fundus \
--Task E  # or A/B/C/... depending on your splitThe **feature extraction** scripts read images from these directories and write `.pickle` feature files under `Final_feature/` or `Final_prediction/`.

---

## Pretrained Weights

Some scripts expect pretrained backbone weights, for example:

- VisionFM: `VFM_Fundus_weights.pth`
- RETFound / FLAIR: see `Feature_Extraction/*/` scripts for how weights are loaded.

Update the corresponding arguments if your weights are in a different location, e.g.:

--pretrained_weights /path/to/your/VFM_Fundus_weights.pth---

## Usage

### 1. Feature Extraction

From the `Github/` root:

# Example: extract RETFound features
python Feature_Extraction/RETFound/RETFound_Feature_Extractor.py \
    --data_path ./dataset/fundus \
    --Task E \
    --output_dir ./Final_featureSimilarly, for FLAIR:

python Feature_Extraction/FLAIR/FLAIR_Feature_Extractor.py \
    --data_path ./dataset/fundus \
    --Task E \
    --output_dir ./Final_featureThese scripts will generate `.pickle` files under `Final_feature/` or `Final_prediction/` which are then consumed by the classification / fusion scripts.

---

### 2. Train Classification Decoders

All classifier scripts share a similar interface. Run them from `Github/`:

#### VisionFM-based classifier

python Classification/VisionFM_based_classifier.py \
    --name visionfm_E \
    --data_path ./dataset/fundus \
    --Task E \
    --epochs 20 \
    --batch_size_per_gpu 512 \
    --output_dir ./results/visionfm#### VisionFM-based classifier

python Classification/RETFound_based_classifier.py \
    --name retfound_E \
    --data_path ./dataset/fundus \
    --Task E \
    --epochs 20 \
    --batch_size_per_gpu 512 \
    --output_dir ./results/retfound#### RETFound-based classifier

python Classification/FLAIR_based_classifier.py \
    --name flair_E \
    --data_path ./dataset/fundus \
    --Task E \
    --epochs 20 \
    --batch_size_per_gpu 512 \
    --output_dir ./results/flair#### FLAIR-based classifier

python Classification/CLIP_based_classifier.py \
    --name clip_E \
    --data_path ./dataset/fundus \
    --Task E \
    --epochs 20 \
    --batch_size_per_gpu 512 \
    --output_dir ./results/clip #### CLIP-based classifier (ViT-L/14)

---

### 3. Mixture of Foundation Models (MIXFound)

Once you have prediction pickles from multiple backbones (VisionFM, RETFound, FLAIR, CLIP), you can run the fusion script:

python MIXFound.py \
    --task G \
    --output_dir ./results/mixfound`MIXFound.py` will:

- Load per‑model prediction `.pickle` files (e.g. `*_pred_G_*.pickle`).
- Compute class‑wise AUC for each model.
- Derive **AUC-based fusion weights** (e.g., power‑scaled and normalized).
- Produce fused predictions and **ROC comparison plots** (e.g. `classification_ROC/G2.png`).
- Log metrics (accuracy, macro AUC, F1, per‑class sensitivity/specificity) to log files under `classification_logs/` or `all_score_classification_logs/`.

Adjust paths and task identifiers inside `MIXFound.py` as needed for your data.

---

## Evaluation & Outputs

The training and evaluation scripts typically produce:

- **Checkpoints**:  
  `results/<name>/checkpoint_<checkpoint_key>_linear.pth`

- **Logs**:  
  `results/<name>/log.txt`  
  `results/<name>/classification_logs/*Task*_*.log`

- **Predictions**:  
  `Final_prediction/<BACKBONE>_pred_<TASK>_<SEED>.pickle`  
  (each entry contains `preds`, `labels`, and `img_path`)

- **Figures**:
  - Confusion matrices: `class_confusion_matrix/<BACKBONE>_Noaug_<TASK>_<SEED>.png`
  - ROC plots for MIXFound: `classification_ROC/<TASK>*.png`

Metrics include:

- Per‑class & mean **Accuracy**
- Per‑class & macro **AUC**
- **Precision / Recall / F1** (optionally skipping background class)
- **Per-class sensitivity & specificity**
- **95% CI for AUC** via bootstrap resampling

---

## Customization

- **Tasks / splits**: controlled via `--Task` (`A`–`G`) and `--data_path`.
- **Backbone choice**: run only the classifier scripts you need, or plug in new feature extractors under `Feature_Extraction/`.
- **Metrics**: extend `Classification/evaluation_funcs.py` or `utils.py` to add custom metrics or logging formats.
- **Paths**: for public release, replace absolute paths with relative ones (e.g. `./dataset/fundus`, `./weights/…`).
