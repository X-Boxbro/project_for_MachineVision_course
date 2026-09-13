# Experiment 2 / 实验2 - Filtering & Edge Detection / 滤波与边缘检测

## Overview / 概述

This experiment covers spatial domain filtering and edge detection techniques. It includes both linear and non-linear filters, as well as a classic differential-based edge detector (Sobel operator).

本实验涵盖空间域滤波和边缘检测技术，包括线性与非线性滤波器，以及经典的差分边缘检测器（Sobel算子）。

---

## Programs / 程序列表

### 1. Median Filter / 中值滤波 (`MedianFilter/`)

**Language:** MATLAB

**Description / 描述:**
Implements a **3×3 median filter** for noise removal. Unlike linear filters (mean/box filter), the median filter is a **nonlinear** operation that:

实现**3×3中值滤波**用于去噪。与线性滤波器（均值/盒滤波器）不同，中值滤波器是**非线性**操作：

1. Extracts the 3×3 neighborhood around each pixel
2. Sorts the 9 pixel values in ascending order
3. Takes the median (5th element) as the output

The median filter is particularly effective at removing **salt-and-pepper noise** while preserving edges better than mean filtering. **Replicate boundary padding** is used.

中值滤波器特别擅长去除**椒盐噪声**，同时比均值滤波更好地保持边缘。使用**复制(replicate)边界填充**。

**Files / 文件:**
- `main.m` — Main script with environment setup, image loading, and result display
- `MyMedfilt2.m` — Core median filter implementation

---

### 2. Edge Detection / 边缘检测 (`EdgeDetection.cpp`)

**Language:** C++ / OpenCV

**Description / 描述:**
Implements the **Sobel operator** for edge detection. The Sobel operator uses two 3×3 convolution kernels:

使用两个3×3卷积核实现**Sobel算子**进行边缘检测：

- **Gx kernel (horizontal gradient):** Detects vertical edges (left-to-right intensity changes)
- **Gy kernel (vertical gradient):** Detects horizontal edges (top-to-bottom intensity changes)

The final edge magnitude is computed as: `G = sqrt(Gx² + Gy²)`

最终边缘强度计算为：`G = sqrt(Gx² + Gy²)`

Boundary handling uses **replicate** mode (edge pixels are clamped to boundary values).

边界处理使用**replicate**模式（边缘像素被夹取到边界值）。

**Files / 文件:**
- `EdgeDetection.cpp` — Complete Sobel edge detection implementation

---

### 3. Histogram Equalization / 直方图均衡化 (`HistogramEqualization.cpp`)

**Language:** C++ / OpenCV

**Description / 描述:**
Implements **histogram equalization** for automatic contrast enhancement. The algorithm:

实现**直方图均衡化**用于自动对比度增强。算法步骤：

1. **Compute histogram / 计算直方图** — Count occurrences of each gray level (0-255)
2. **Compute cumulative histogram (CDF) / 计算累计直方图** — Cumulative sum of histogram
3. **Build lookup table (LUT) / 构建查找表** — `map[k] = round(255 * CDF[k] / totalPixels)`
4. **Apply mapping / 应用映射** — Replace each pixel value using the LUT

Result: The output image has a more uniform histogram distribution, enhancing overall contrast.

结果：输出图像的直方图分布更加均匀，增强了整体对比度。

**Files / 文件:**
- `HistogramEqualization.cpp` — Complete histogram equalization implementation

---

### 4. Linear Filter / 线性滤波 (`LinearFilter.cpp`)

**Language:** C++ / OpenCV

**Description / 描述:**
Implements a **3×3 box filter (mean filter)** — the simplest linear filter. The kernel uses equal weights:

实现**3×3盒滤波器（均值滤波器）**——最简单的线性滤波器。核使用相等的权重：

```
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

Each pixel is replaced by the arithmetic mean of its 3×3 neighborhood. This smooths the image and reduces noise, but also causes blurring (loss of detail).

每个像素被替换为其3×3邻域的算术平均值。这能平滑图像并减少噪声，但也会导致模糊（细节丢失）。

**Files / 文件:**
- `LinearFilter.cpp` — Complete box filter implementation

---

## Notes / 注意事项

- All programs read `testimg.jpg` as input (replaceable)
- 所有程序以`testimg.jpg`为输入（可更换）
- Boundary handling: replicate mode for both C++ and MATLAB implementations
- 边界处理：C++和MATLAB实现均使用replicate模式
