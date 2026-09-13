# Course Design - Hand Gesture Recognition

中文版：[README.md](README.md)

---

## Overview

This is the comprehensive course design project for the Machine Vision course, divided into two parts:

- **Part 1:** Traditional machine learning approach using SVM with HOG features for hand pose classification (4 classes)
- **Part 2:** Deep learning approach using fine-tuned ConvNeXt-Tiny for gesture recognition (6 classes)

---

## Part 1 - SVM Hand Pose Recognition (C++/OpenCV)

**Author:** Xie Gang | **Student ID:** 2411628

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
├── hand_pose_inference.cpp      # Main inference program
├── hand_pose_svm_model.yml      # SVM model file
├── hand_pose_svm_model_params.yml  # PCA parameters
└── requirements.txt             # Environment setup guide
```

### Requirements

- OpenCV ≥ 4.0 (core, imgproc, imgcodecs, ml)
- C++11 or higher
- Compatible with Windows (MinGW/MSVC) and Linux

### Usage

1. Compile with OpenCV: `g++ -o hand_pose_inference hand_pose_inference.cpp ...`
2. Place model files in the same directory
3. Update `TEST_FOLDER` path in source code
4. Run: `./hand_pose_inference`

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
├── hand_gesture_test.py         # Main test script
├── best_model.pth               # Trained model weights (111MB)
└── requirements.txt             # Python dependencies
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
2. Update `DATA_DIR` in the script to your dataset path
3. Run: `python hand_gesture_test.py`

---

## Notes

- Part 1 model files (`*_svm_model.yml`, `*_svm_model_params.yml`) are in `.gitignore` due to large size
- Part 2 `best_model.pth` (111MB) exceeds GitHub's 100MB file limit and is in `.gitignore`
- These files should be downloaded or generated separately
