# MIXFound

**MIXFound** provides a complete pipeline for **fundus image classification** based on several powerful visual backbones (**VisionFM**, **RETFound**, **FLAIR**, **CLIP**) and a novel multi-model fusion strategy.

---

## 📑 Table of Contents

- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Environment Setup](#-environment-setup)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
  - [1. Feature Extraction](#1-feature-extraction)
  - [2. Train Classification Decoders](#2-train-classification-decoders)
  - [3. Multi-Model Fusion (MIXFound)](#3-multi-model-fusion-mixfound)
- [Evaluation & Outputs](#-evaluation--outputs)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Features

- **🧩 Multiple Backbones Supported**
  - **VisionFM** (Fundus-specific foundation model)
  - **RETFound**
  - **FLAIR**
  - **OpenAI CLIP** (ViT-L/14)

- **🎛 Flexible Classification Decoders**
  - Linear heads (`ClsHead`) on frozen features.
  - Full support for multi-class classification tasks.

- **📈 Rich Evaluation Metrics**
  - Accuracy, Precision, Recall, F1-Score.
  - Class-wise and Macro **ROC-AUC**.
  - **Bootstrap 95% CI** for AUC.
  - Confusion matrices and per-class sensitivity/specificity.

- **🚀 Multi-Model Fusion (MIXFound)**
  - Adaptive AUC-based weighting across multiple backbones.
  - Automated generation of ROC comparison plots.

- **⚡ Reproducible & Scalable**
  - Configurable via `argparse` (Tasks A–G, seeds, etc.).
  - Distributed Data Parallel (DDP) support.

---

## 📂 Repository Structure

```text
Github/
├── environment.yml                # Conda environment specification
├── MIXFound.py                    # 🚀 Main Script: Multi-model fusion
├── utils.py                       # Shared utilities (DDP, metrics, logging)
├── dataset/                       # Dataset utilities / wrappers
│
├── Classification/                # 🧠 Linear Probing / Decoders
│   ├── CLIP_based_classifier.py
│   ├── FLAIR_based_classifier.py
│   ├── RETFound_based_classifier.py
│   ├── VisionFM_based_classifier.py
│   ├── utils.py
│   └── evaluation_funcs.py
│
└── Feature_Extraction/            # 📸 Feature Extractors
    ├── RETFound/
    │   ├── RETFound_Feature_Extractor.py
    │   └── models_vit.py
    ├── FLAIR/
    │   └── FLAIR_Feature_Extractor.py
    └── ...
---

## ⚙️ Environment Setup
# 1. Create environment from provided file
conda env create -f environment.yml -n visionfm

# 2. Activate environment
conda activate visionfm

# 3. (Optional) Export requirements
pip freeze > requirements.txt

## 💾 Data Preparation

dataset/
└── fundus/
    ├── A/.../G/                   # Task Splits (e.g., Task A, Task E)
    │   ├── AMD/                   # Class folder (Disease Name)
    │   ├── Glaucoma/              # Class folder
    │   ├── High myopia/           # Class folder
    │   └── Normal fundus/         # Class folder
    │
    ├── training/
    │   └── training_labels.txt    # Format: path;label
    │   └── test_labels/           # Validation/Test images
    └── evaluation/
        └── evaluation_labels.txt  # Format: path;label
## 🚀 Usage

# 1. Feature Extraction

Extract features to .pickle files before training classifiers.

# RETFound Example:
python Feature_Extraction/RETFound/RETFound_Feature_Extractor.py \
    --data_path ./dataset/fundus \
    --Task E \
    --output_dir ./Final_feature

