# Course Design - Hand Gesture Recognition

中文版：[README.md](README.md)

---

## Overview

This is the comprehensive course design project for the Machine Vision course, divided into two parts:

- **Part 1:** Traditional machine learning approach using SVM with HOG features for hand pose classification (4 classes)
- **Part 2:** Deep learning approach using fine-tuned ConvNeXt-Tiny for gesture recognition (6 classes)

---

## Part 1 - SVM Hand Pose Recognition (C++/OpenCV)

### Description

A hand pose recognition system based on **Support Vector Machine (SVM)** with **Histogram of Oriented Gradients (HOG)** features and **PCA** dimensionality reduction.

### Key Features

| Feature | Details |
|---------|---------|
| **Algorithm** | SVM (RBF kernel) + PCA |
| **Feature** | HOG (128×128 window, 16×16 block, 8×8 cell, 9 bins) |
| **Classes** | A, C, Five, V (4 classes) |
| **Preprocessing** | Gamma correction → CLAHE → Gaussian blur → OTSU thresholding → Morphological closing |

### File Structure

```
Part1_SVM_HandPose/
├── hand_pose_train.cpp          # Training code (SVM+HOG feature extraction & training)
├── hand_pose_inference.cpp      # Inference/prediction code
├── hand_pose_svm_model.yml      # Trained SVM model
├── hand_pose_svm_model_params.yml  # PCA normalization parameters
└── requirements.txt             # Environment setup guide
```

### Training Code

`hand_pose_train.cpp` is the core training code for Part 1, including:
- HOG + LBP + Hu moments + contour feature fusion
- PCA dimensionality reduction
- K-fold cross-validation for automatic best SVM model selection
- Data augmentation (brightness, flipping, rotation, scaling, Gaussian noise)

> Note: The code is currently commented out (due to compilation environment dependencies). Uncomment and configure the dataset path to compile and run.

### Requirements

- OpenCV ≥ 4.0 (core, imgproc, imgcodecs, ml)
- C++11 or higher
- Compatible with Windows (MinGW/MSVC) and Linux

### Usage

1. Compile with OpenCV: `g++ -o hand_pose_train hand_pose_train.cpp ...`
2. Place model files in the same directory
3. Update `TRAIN_ROOT` path in source code
4. Run: `./hand_pose_train`

---

## Part 2 - ConvNeXt-Tiny Deep Learning (Python/PyTorch)

### Description

A deep learning-based gesture recognition system using **ConvNeXt-Tiny** architecture fine-tuned on ImageNet-1K pretrained weights.

### Key Features

| Feature | Details |
|---------|---------|
| **Architecture** | ConvNeXt-Tiny |
| **Classes** | A, B, C, Five, Point, V (6 classes) |
| **Preprocessing** | Resize 224×224 → Normalize (ImageNet stats) |
| **Validation Accuracy** | 100% |
| **Test Accuracy** | 99.72% |

### File Structure

```
Part2_DeepLearning_HandGesture/
├── train.py                     # Training code (ConvNeXt-Tiny fine-tuning)
├── dataset.py                   # Dataset loading and augmentation
├── hand_gesture_test.py         # Testing/inference code
├── training_history.txt         # Training history (30 epochs)
├── best_model.pth               # Trained model weights (111MB)
└── requirements.txt             # Python dependencies
```

### Training Code

`train.py` is the core training code for Part 2, including:
- ConvNeXt-Tiny model building (ImageNet pretrained weights)
- AdamW optimizer + Cosine annealing learning rate scheduling
- Train/validation split (80/20 stratified sampling)
- Automatic best model saving
- Training history logging

Training dataset structure:
```
Hand_Posture_Hard_Stu/
├── A/         # Class A gesture images
├── B/         # Class B gesture images
├── C/         # Class C gesture images
├── Five/      # Class Five gesture images
├── Point/     # Class Point gesture images
└── V/         # Class V gesture images
```

### Requirements

```
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
numpy>=1.20.0
```

### Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Train: `python train.py --data_root <dataset_path>`
3. Test: `python hand_gesture_test.py`

---

## Notes

- Part 1 model files (`*_svm_model.yml`, `*_svm_model_params.yml`) are in `.gitignore` due to large size
- Part 2 `best_model.pth` (111MB) exceeds GitHub's 100MB file limit and is in `.gitignore`
- These files should be downloaded or generated separately
