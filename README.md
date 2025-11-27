# Root Segmentation Beyond Species Boundaries: Generalizable Framework for Anatomical Analysis

[![Dataset](https://img.shields.io/badge/Data-Zenodo-blue)](https://doi.org/10.5281/zenodo.17726414)
[![Models](https://img.shields.io/badge/Models-Zenodo-green)](https://doi.org/10.5281/zenodo.17737703)

## Overview

This repository contains the official implementation of **"Root Segmentation Beyond Species Boundaries: Generalizable Framework for Anatomical Analysis."** This project provides a deep learning framework for the anatomical analysis of agricultural crops, specifically focusing on **Pearl Millet** and **Sorghum**. It includes generalized models for:
1.  Binary Segmentation
2.  Multiclass Segmentation


## Data & Models

The dataset and pre-trained models are hosted on Zenodo:

* **Dataset:** [Download here](https://doi.org/10.5281/zenodo.17726414)
* **Models:** [Download here](https://doi.org/10.5281/zenodo.17737703)

## Installation

### Prerequisites


# Clone the repository
git clone [https://github.com/janetkok/Root-Segmentation-Beyond-Species-Boundaries.git](https://github.com/janetkok/Root-Segmentation-Beyond-Species-Boundaries.git)
cd Root-Segmentation-Beyond-Species-Boundaries

# Install dependencies
pip install torch torchvision numpy opencv-python pillow tqdm natsort matplotlib


## Directory Structure

To run the training and testing scripts, organize your downloaded data as follows:

data_root/
├── binary/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── labels/
└── multiclass/
    ├── images/
    └── labels/


## Usage: Binary Segmentation

The binary segmentation module is located in the `binary/` folder. It uses edge-enhanced input channels.

### 1\. Data Preparation

Before training or testing, you must generate the pre-processed input features (gradient and gaussian maps).

```bash
# Navigate to the binary folder and update paths in process.py if necessary
python binary/process.py
```

*This script generates `images_R`, `images_G`, and `images_B` folders containing Sobel gradients and adaptive thresholds.*

### 2\. Training

To train the binary model (e.g., UNet):

```bash
python binary/train.py \
  --name='baseline_cross_entropy' \
  --model-type='unet' \
  --data-dir='/path/to/your/binary_data' \
  --max-epoch=500 \
  --batch-size=32 \
  --crop-size=512 \
  --num-cls=2 \
  --lr=1e-4 \
  --val-start=100 \
  --seed=15 \
  --val-resiz=1024 \
  --weight-decay=1e-3 \
```

### 3\. Testing

To test the binary model:

```bash
python binary/test.py
```

*Note: Ensure the model weights path inside `test.py` points to your trained model.*

-----

## Usage: Multiclass Segmentation

The multiclass segmentation module is located in the `paint/` folder.

### 1\. Data Preparation

Generate the robust gradient orientation maps required for the multiclass inputs.

```bash
# Update paths in prepare.py to point to your multiclass images
python paint/prepare.py
```

*This script generates `images_gradient3` and `images_gradient7`.*

### 2\. Training

The multiclass training requires text files defining the train/val splits (e.g., `train_millet.txt`, `val_millet.txt`).

```bash
python paint/train.py \
  --name='unet_ce_millet' \
  --model-type='unet' \
  --data-dir='/path/to/your/multiclass_data' \
  --train_path='./paint/train_millet.txt' \
  --val_path='./paint/val_millet.txt' \
  --max-epoch=500 \
  --batch-size=8 \
  --crop-size=768 \
  --num-cls=9 \
  --lr=1e-4
  --seed = 1 \
  --val-start=50 \
  --val-resize=1024 \
  --weight-decay=1e-3
```

### 3\. Testing

Run the evaluation script specifying the data paths and the binary mask path (used for background filtering).

```bash
python paint/test.py \
  --binary_path 'images_binary' \
  --data_path '/path/to/your/multiclass/images' \
  --predict_path '/path/to/save/predictions' \
  --model_path '/path/to/your/best_model.pth'
```


