# Machine Vision Course Projects

中文版：[README.md](README.md)

---

## Overview

This repository contains all lab experiments and course design projects for the **Machine Vision** course. The projects cover fundamental image processing techniques, morphological operations, thresholding segmentation, edge detection, filtering, and deep learning-based gesture recognition.

---

## Repository Structure

```
project_repo/
├── CourseDesign/          # Course Design - Hand Gesture Recognition
│   ├── Part1_SVM_HandPose/        # Part 1: SVM-based Hand Pose Recognition
│   └── Part2_DeepLearning_HandGesture/  # Part 2: ConvNeXt Deep Learning Gesture Recognition
├── Experiment1/           # Experiment 1 - Basic Image Processing (MATLAB + C++/OpenCV)
├── Experiment2/           # Experiment 2 - Filtering & Edge Detection (MATLAB + C++/OpenCV)
└── Experiment3/           # Experiment 3 - Thresholding & Contour Detection (MATLAB + C++/OpenCV)
```

---

## Experiment Descriptions

### Experiment 1 - Basic Image Processing

Covers fundamental image processing operations including distance transform, histogram computation and display, convolution (both spatial and FFT-based), and integral image computation.

- **Distance Transform** — Implements chamfer distance transform algorithm (MATLAB)
- **Histogram Display** — Manually computes and visualizes grayscale histogram (C++/OpenCV)
- **Convolution** — Gaussian convolution using both direct spatial computation and FFT-based method (C++/OpenCV)
- **Integral Image** — Computes summed-area table using recursive formula (MATLAB)

> Details: [Experiment1/README_en.md](Experiment1/README_en.md)

### Experiment 2 - Filtering & Edge Detection

Covers median filtering, Sobel edge detection, histogram equalization, and linear (box) filtering operations.

- **Median Filter** — 3×3 median filter with replicate boundary padding (MATLAB)
- **Edge Detection** — Sobel operator for edge detection (C++/OpenCV)
- **Histogram Equalization** — Histogram equalization for contrast enhancement (C++/OpenCV)
- **Linear Filter** — 3×3 box filter / mean filter (C++/OpenCV)

> Details: [Experiment2/README_en.md](Experiment2/README_en.md)

### Experiment 3 - Thresholding & Contour Detection

Covers OTSU thresholding, iterative optimal threshold selection, P-percent thresholding, and inner contour detection using morphological operations.

- **OTSU Threshold** — OTSU (Otsu's method) automatic thresholding (MATLAB)
- **Iterative Optimal Threshold** — Iterative threshold convergence algorithm (MATLAB)
- **P-Percent Threshold** — P-percent adaptive thresholding (C++/OpenCV)
- **Inner Contour Detection** — Detects contours using morphological erosion (C++/OpenCV)

> Details: [Experiment3/README_en.md](Experiment3/README_en.md)

### Course Design - Hand Gesture Recognition

A two-part comprehensive project: Part 1 uses SVM with HOG features for hand pose classification (4 classes); Part 2 uses a fine-tuned ConvNeXt-Tiny deep learning model for 6-class gesture recognition.

- **Part 1** — SVM + HOG feature-based hand pose recognition (C++/OpenCV)
- **Part 2** — ConvNeXt-Tiny deep learning gesture recognition (Python/PyTorch)

> Details: [CourseDesign/README_en.md](CourseDesign/README_en.md)

---

## Technical Stack

| Component | Language | Libraries / Tools |
|-----------|----------|-------------------|
| Experiments 1-3 (MATLAB parts) | MATLAB | Image Processing Toolbox |
| Experiments 1-3 (C++ parts) | C++ | OpenCV 4.x |
| Course Design Part 1 | C++ | OpenCV 4.x, SVM |
| Course Design Part 2 | Python 3 | PyTorch, torchvision, PIL |
