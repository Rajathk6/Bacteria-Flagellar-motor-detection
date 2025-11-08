# Tomo-Detect Research Pipeline

A modular deep learning framework for **localizing bacterial flagellar motors** in Cryo-Electron Tomography (Cryo-ET) volumes.  
This repository provides standalone scripts for preprocessing, model training, inference, and postprocessing.

For command-line usage, refer to the **[Tomo-Detect CLI on PyPI](https://pypi.org/project/tomo-detect/0.1.7/)**  
and the **[Tomo-Detect CLI GitHub repository](https://github.com/Rajathk6/tomo-detect-cli)**.

## System Requirements

 - Python 3.8 or higher

 - PyTorch ≥ 2.0

 - CUDA-capable GPU (recommended)

 - Minimum 8 GB RAM

 - 2 GB disk space for models

---

## Dataset

The dataset and the pre-trained models for this project is **not included**.  
Download the dataset directly from the official competition:

> [BYU Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/data)


Download the pre-trained weights for the project from the google cloud bucket or Drive

> [Pre-trained Weights GCP Bucket](https://storage.cloud.google.com/pre-trained-model-tomo-detect/r3d200_704_350984_epoch400.pt)

> [Pre-trained Weights Drive](https://drive.google.com/drive/folders/15l621DVxajvY02v6aJmCoAsOeW3eix-r?usp=sharing)


### Place the data under a suitable directory, e.g.:

```bash
tomo-detect/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── models/
│   ├── checkpoints/
│   └── best_model.pth
│
├── scripts/
│   ├── standalone_preprocess.py
│   ├── standalone_train.py
│   ├── standalone_inference.py
│   ├── standalone_postprocess.py
│   └── model_visualization.py
│
└── README.md
```

## File Descriptions

### standalone_preprocess.py

- Loads raw Cryo-ET tomograms (.npy, .jpg, .mrc, or folders of slices).

* Normalizes and resizes 3D volumes to a fixed size.

+ Generates Gaussian heatmap labels using coordinates from labels_new.csv.

- Saves processed tensors under data/processed/.

### standalone_train.py

- Defines and trains a 3D CNN for flagellar motor localization.

- Handles data loading, augmentation, checkpoint saving, and validation.

- Supports GPU acceleration and hyperparameter customization.

### standalone_inference.py

- Performs 3D inference on new tomograms using trained models.

- Generates probability maps and binary masks of detected regions.

- Supports both CPU and GPU backends.

### standalone_postprocess.py

- Converts raw probability maps into discrete motor coordinates.

- Applies filtering and thresholding to refine predictions.

- Outputs results as .csv, .npy, and .json summaries.

### model_visualization.py

- Provides visualization utilities for inspection of model outputs.

- Supports 2D slice overlays and 3D volume visualizations.
