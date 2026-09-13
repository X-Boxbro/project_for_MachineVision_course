# Course Design / 课程设计 - Hand Gesture Recognition / 手势识别

## Overview / 概述

This is the comprehensive course design project for the Machine Vision course, divided into two parts:

本课程设计是机器视觉课程的综合性项目，分为两个部分：

- **Part 1:** Traditional machine learning approach using SVM with HOG features for hand pose classification (4 classes)
- **Part 2:** Deep learning approach using fine-tuned ConvNeXt-Tiny for gesture recognition (6 classes)

- **第一部分：** 使用SVM结合HOG特征的传统机器学习方法进行手姿态分类（4类）
- **第二部分：** 使用微调ConvNeXt-Tiny深度学习模型进行手势识别（6类）

---

## Part 1 / 第一部分 - SVM Hand Pose Recognition (C++/OpenCV)

**Author:** 谢刚 (Xie Gang) | **Student ID:** 2411628

### Description / 描述

A hand pose recognition system based on **Support Vector Machine (SVM)** with **Histogram of Oriented Gradients (HOG)** features and **PCA** dimensionality reduction.

基于**支持向量机(SVM)**结合**方向梯度直方图(HOG)**特征和**PCA**降维的手姿态识别系统。

### Key Features / 主要特点

| Feature | Details |
|---------|---------|
| **Algorithm** | SVM (RBF kernel) + PCA |
| **Feature** | HOG (128×128 window, 16×16 block, 8×8 cell, 9 bins) |
| **Classes / 类别** | A, C, Five, V (4 classes) |
| **Preprocessing** | Gamma correction → CLAHE → Gaussian blur → OTSU thresholding → Morphological closing |

### File Structure / 文件结构

```
Part1_SVM_HandPose/
├── hand_pose_inference.cpp      # Main inference program / 主推理程序
├── hand_pose_svm_model.yml      # SVM model file / SVM模型文件
├── hand_pose_svm_model_params.yml  # PCA parameters / PCA参数
└── requirements.txt             # Environment setup guide / 环境配置说明
```

### Requirements / 环境要求

- OpenCV ≥ 4.0 (core, imgproc, imgcodecs, ml)
- C++11 or higher
- Compatible with Windows (MinGW/MSVC) and Linux

### Usage / 使用方法

1. Compile with OpenCV: `g++ -o hand_pose_inference hand_pose_inference.cpp ...`
2. Place model files in the same directory
3. Update `TEST_FOLDER` path in source code
4. Run: `./hand_pose_inference`

---

## Part 2 / 第二部分 - ConvNeXt-Tiny Deep Learning (Python/PyTorch)

### Description / 描述

A deep learning-based gesture recognition system using **ConvNeXt-Tiny** architecture fine-tuned on ImageNet-1K pretrained weights.

使用**ConvNeXt-Tiny**架构、基于ImageNet-1K预训练权重微调的深度学习手势识别系统。

### Key Features / 主要特点

| Feature | Details |
|---------|---------|
| **Architecture** | ConvNeXt-Tiny |
| **Classes / 类别** | A, B, C, Five, Point, V (6 classes) |
| **Preprocessing** | Resize 224×224 → Normalize (ImageNet stats) |
| **Validation Accuracy / 验证集准确率** | 100% |
| **Test Accuracy / 测试集准确率** | 99.72% |

### File Structure / 文件结构

```
Part2_DeepLearning_HandGesture/
├── hand_gesture_test.py         # Main test script / 主测试脚本
├── best_model.pth               # Trained model weights (111MB) / 训练好的模型权重
└── requirements.txt             # Python dependencies / Python依赖
```

### Requirements / 环境要求

```
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
numpy>=1.20.0
```

### Usage / 使用方法

1. Install dependencies: `pip install -r requirements.txt`
2. Update `DATA_DIR` in the script to your dataset path
3. Run: `python hand_gesture_test.py`

---

## Notes / 注意事项

- Part 1 model files (`*_svm_model.yml`, `*_svm_model_params.yml`) are in `.gitignore` due to large size
- Part 2 `best_model.pth` (111MB) exceeds GitHub's 100MB file limit and is in `.gitignore`
- These files should be downloaded or generated separately
- 第一部分模型文件因体积较大已在.gitignore中
- 第二部分模型文件(111MB)超过GitHub 100MB限制，已在.gitignore中
