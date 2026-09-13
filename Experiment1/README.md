# Experiment 1 / 实验1 - Basic Image Processing / 基础图像处理

## Overview / 概述

This experiment covers four fundamental image processing operations, implemented in both MATLAB and C++/OpenCV. These operations form the foundation of many computer vision algorithms.

本实验涵盖四种基本图像处理操作，使用MATLAB和C++/OpenCV实现。这些操作构成了许多计算机视觉算法的基础。

---

## Programs / 程序列表

### 1. Distance Transform / 距离变换 (`DistanceTransform/`)

**Language:** MATLAB

**Description / 描述:**
The distance transform computes the distance from each pixel to the nearest foreground (non-zero) pixel in a binary image. This implementation uses the **chamfer distance transform** algorithm, which performs two passes over the image:

距离变换计算每个像素到二值图像中最近前景（非零）像素的距离。本实现使用**Chamfer距离变换**算法，对图像进行两次扫描：

- **First pass (top-left to bottom-right):** Checks top and left neighbors
- **Second pass (bottom-right to top-left):** Checks bottom and right neighbors, updates minimum

**Files / 文件:**
- `main.m` — Main script, reads image, converts to binary, calls MyDisTrans
- `MyDisTrans.m` — Core algorithm implementation

---

### 2. Histogram Display / 直方图显示 (`HistogramDisplay.cpp`)

**Language:** C++ / OpenCV

**Description / 描述:**
Manually computes and visualizes the grayscale histogram of an image. Instead of using OpenCV's built-in `calcHist` and `normalize` functions, this program:

手动计算并可视化图像的灰度直方图。程序未使用OpenCV内置的`calcHist`和`normalize`函数，而是：

1. Iterates through all pixels to count occurrences of each gray level (0-255)
2. Normalizes the histogram to fit the display canvas
3. Draws the histogram as a bar chart using `rectangle()`

**Files / 文件:**
- `HistogramDisplay.cpp` — Complete implementation

---

### 3. Convolution / 卷积 (`Convolution.cpp`)

**Language:** C++ / OpenCV

**Description / 描述:**
Implements Gaussian convolution using two different methods:

使用两种不同方法实现高斯卷积：

1. **Direct spatial convolution** — Applies a 128×128 Gaussian kernel using nested loops (4-layer) with border skipping (BORDER_CONSTANT)
2. **FFT-based convolution** — Pads image and kernel to optimal DFT size, performs DFT, multiplies in frequency domain, then inverse DFT to obtain result

The Gaussian kernel is generated with σ=20.0 and normalized so that all weights sum to 1.

高斯核使用σ=20.0生成，并归一化使所有权值之和为1。

**Files / 文件:**
- `Convolution.cpp` — Both implementations (spatial and FFT-based)

---

### 4. Integral Image / 积分图像 (`IntegralImage/`)

**Language:** MATLAB

**Description / 描述:**
Computes the **summed-area table** (integral image) of a grayscale image. The integral image at position (i,j) contains the sum of all pixel values above and to the left:

计算灰度图像的**积分图**（求和面积表）。位置(i,j)处的积分图像包含该位置上方和左侧所有像素值的总和：

```
SA(i,j) = img(i,j) + SA(i-1,j) + SA(i,j-1) - SA(i-1,j-1)
```

This recursive formula allows O(1) computation of any rectangular region sum after O(M×N) preprocessing.

该递归公式在O(M×N)预处理后，可以实现任意矩形区域和的O(1)计算。

**Files / 文件:**
- `main.m` — Main script
- `MyInteImg.m` — Core algorithm implementation

---

## Notes / 注意事项

- Input images can be freely replaced / 输入图像可随意更换
- In MATLAB, computed matrix data cannot be directly displayed as an image — type conversion (e.g., `uint8()`) may be required
- In MATLAB中，运算得到的矩阵数据不能直接输出为图像——可能需要进行类型转换（如`uint8()`）
