# MIXFound: A mixture of ophthalmic foundation models enables mitigation of altitude bias

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Fundus_Classification-green)]()

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
- [References & Acknowledgements](#-references--acknowledgements)
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

```

---

## ⚙️ Environment Setup

> ⚠️ **Important Note**: This repository is designed to be executed from the `Github/` root directory. Please ensure you are in the root folder before running any scripts.

We recommend using **conda** for environment management:

```bash
# 1. Create environment from provided file
conda env create -f environment.yml -n visionfm

# 2. Activate environment
conda activate visionfm

# 3. (Optional) Export requirements for pip users
pip freeze > requirements.txt

```

---

## 💾 Data Preparation

The classification scripts expect the fundus dataset organized as follows:

```text
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

```

> **Note**: Pretrained weights (e.g., `VFM_Fundus_weights.pth`) should be placed in the appropriate paths or specified via arguments.

---

## 🚀 Usage

### 1. Feature Extraction

Extract features to `.pickle` files before training classifiers.

**RETFound Example:**

```bash
python Feature_Extraction/RETFound/RETFound_Feature_Extractor.py \
    --data_path ./dataset/fundus \
    --Task E \
    --output_dir ./Final_feature

```

**FLAIR Example:**

```bash
python Feature_Extraction/FLAIR/FLAIR_Feature_Extractor.py \
    --data_path ./dataset/fundus \
    --Task E \
    --output_dir ./Final_feature

```

### 2. Train Classification Decoders

Train linear decoders on the extracted features.

**VisionFM Classifier:**

```bash
python Classification/VisionFM_based_classifier.py \
    --name visionfm_E \
    --data_path ./dataset/fundus \
    --Task E \
    --epochs 20 \
    --batch_size_per_gpu 512 \
    --output_dir ./results/visionfm

```

**RETFound Classifier:**

```bash
python Classification/RETFound_based_classifier.py \
    --name retfound_E \
    --data_path ./dataset/fundus \
    --Task E \
    --epochs 20 \
    --batch_size_per_gpu 512 \
    --output_dir ./results/retfound

```

*(Scripts for `FLAIR` and `CLIP` follow the same usage pattern.)*

### 3. Multi-Model Fusion (MIXFound)

Once you have prediction pickles from multiple backbones, run the fusion script:

```bash
python MIXFound.py \
    --task G \
    --output_dir ./results/mixfound

```

**Workflow:**

1. Loads per-model prediction `.pickle` files.
2. Computes class-wise AUC for each model.
3. Derives **AUC-based fusion weights**.
4. Produces fused predictions and ROC comparison plots.

---

## 📊 Evaluation & Outputs

The pipeline generates the following artifacts:

| Output Type | Path Pattern | Description |
| --- | --- | --- |
| **Checkpoints** | `results/<name>/checkpoint_*.pth` | Saved model weights. |
| **Logs** | `results/<name>/log.txt` | Training logs. |
| **Predictions** | `Final_prediction/*_pred_*.pickle` | Contains `preds`, `labels`, and `img_path`. |
| **Confusion Matrix** | `class_confusion_matrix/*.png` | Visualized confusion matrices per task/seed. |
| **ROC Plots** | `classification_ROC/*.png` | Comparative ROC curves (Individual Models vs. MIXFound). |

---

## 📚 References & Acknowledgements

We gratefully acknowledge the authors of the following foundation models for their open-source contributions. This project builds upon these excellent works:

<details>
<summary><strong>Click to expand BibTeX</strong></summary>

```bibtex
% CLIP
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}

% RETFound
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yuyin and Chia, Mark A and Wagner, S and Ayhan, Murat S and Williamson, DJ and Struyven, RR and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

% FLAIR (Please verify the specific FLAIR paper reference)
@article{silva2023flair,
  title={FLAIR: Foundation Language And Image representations},
  author={Silva-Rodriguez, Julio and others},
  journal={arXiv preprint},
  year={2023}
}

% VisionFM (Please replace with the official citation)
@article{visionfm2024,
  title={VisionFM: A Foundation Model for Fundus...},
  author={...},
  journal={...},
  year={2024}
}

```

</details>

* **CLIP**: [OpenAI](https://github.com/openai/CLIP)
* **RETFound**: [Nature Article](https://www.nature.com/articles/s41586-023-06555-x)
* **FLAIR**: [Project Link / Paper]
* **VisionFM**: [Project Link / Paper]

---

## 📝 Citation

If you find this project useful for your research, please cite:

```bibtex
@article{YourCitation2025,
  title={MIXFound: A Mixture of Foundation Models...},
  author={...},
  journal={...},
  year={2025}
}

```

## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
