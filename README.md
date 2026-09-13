# Machine Vision Course Projects / 机器视觉课程项目

## Overview / 项目简介

This repository contains all lab experiments and course design projects for the **Machine Vision** course. The projects cover fundamental image processing techniques, morphological operations, thresholding segmentation, edge detection, filtering, and deep learning-based gesture recognition.

本仓库包含**机器视觉**课程的全部实验和课程设计项目。项目涵盖基本图像处理技术、形态学操作、阈值分割、边缘检测、滤波以及基于深度学习的手势识别。

---

## Repository Structure / 目录结构

```
project_repo/
├── CourseDesign/          # 课程设计 - Hand Pose & Gesture Recognition
│   ├── Part1_SVM_HandPose/        # Part 1: SVM-based Hand Pose Recognition
│   └── Part2_DeepLearning_HandGesture/  # Part 2: ConvNeXt Deep Learning Gesture Recognition
├── Experiment1/           # 实验1 - Basic Image Processing (MATLAB + C++/OpenCV)
├── Experiment2/           # 实验2 - Filtering & Edge Detection (MATLAB + C++/OpenCV)
└── Experiment3/           # 实验3 - Thresholding & Contour Detection (MATLAB + C++/OpenCV)
```

---

## Experiment Descriptions / 实验说明

### Experiment 1 / 实验1 - Basic Image Processing / 基础图像处理

Covers fundamental image processing operations including distance transform, histogram computation and display, convolution (both spatial and FFT-based), and integral image computation.

涵盖基本图像处理操作，包括距离变换、直方图计算与显示、卷积（空间域和基于FFT）以及积分图像计算。

- **Distance Transform / 距离变换** — Implements chamfer distance transform algorithm in MATLAB
- **Histogram Display / 直方图显示** — Manually computes and visualizes grayscale histogram in C++/OpenCV
- **Convolution / 卷积** — Implements Gaussian convolution using both direct spatial computation and FFT-based method in C++/OpenCV
- **Integral Image / 积分图像** — Computes summed-area table using recursive formula in MATLAB

### Experiment 2 / 实验2 - Filtering & Edge Detection / 滤波与边缘检测

Covers median filtering, Sobel edge detection, histogram equalization, and linear (box) filtering operations.

涵盖中值滤波、Sobel边缘检测、直方图均衡化和线性（均值）滤波操作。

- **Median Filter / 中值滤波** — Implements 3×3 median filter with replicate boundary padding in MATLAB
- **Edge Detection / 边缘检测** — Implements Sobel operator for edge detection in C++/OpenCV
- **Histogram Equalization / 直方图均衡化** — Implements histogram equalization for contrast enhancement in C++/OpenCV
- **Linear Filter / 线性滤波** — Implements 3×3 box filter (mean filter) in C++/OpenCV

### Experiment 3 / 实验3 - Thresholding & Contour Detection / 阈值分割与轮廓检测

Covers OTSU thresholding, iterative optimal threshold selection, P-percent thresholding, and inner contour detection using morphological operations.

涵盖OTSU阈值法、迭代最优阈值选择、P率阈值化以及基于形态学操作的内边界检测。

- **OTSU Threshold / OTSU阈值** — Implements OTSU (Otsu's method) automatic thresholding in MATLAB
- **Iterative Optimal Threshold / 迭代最优阈值** — Implements iterative threshold convergence algorithm in MATLAB
- **P-Percent Threshold / P率阈值化** — Implements P-percent adaptive thresholding in C++/OpenCV
- **Inner Contour Detection / 内边界跟踪** — Detects contours using morphological erosion in C++/OpenCV

### Course Design / 课程设计 - Hand Gesture Recognition / 手势识别

A two-part comprehensive project: Part 1 uses SVM with HOG features for hand pose classification (4 classes); Part 2 uses a fine-tuned ConvNeXt-Tiny deep learning model for 6-class gesture recognition.

分为两部分的综合性项目：第一部分使用SVM结合HOG特征进行手姿态分类（4类）；第二部分使用微调ConvNeXt-Tiny深度学习模型进行6类手势识别。

- **Part 1 / 第一部分** — SVM + HOG feature-based hand pose recognition (C++/OpenCV)
- **Part 2 / 第二部分** — ConvNeXt-Tiny deep learning gesture recognition (Python/PyTorch)

---

## Technical Stack / 技术栈

| Component | Language | Libraries / Tools |
|-----------|----------|-------------------|
| Experiments 1-3 (MATLAB parts) | MATLAB | Image Processing Toolbox |
| Experiments 1-3 (C++ parts) | C++ | OpenCV 4.x |
| Course Design Part 1 | C++ | OpenCV 4.x, SVM |
| Course Design Part 2 | Python 3 | PyTorch, torchvision, PIL |

---

## Student Information / 学生信息

- **Name / 姓名:** 谢刚 (Xie Gang)
- **Student ID / 学号:** 2411628
- **Course / 课程:** 机器视觉 (Machine Vision)
- **Semester / 学期:** 大二下 (Second Year, Second Semester)
