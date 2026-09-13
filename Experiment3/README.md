# Experiment 3 / 实验3 - Thresholding & Contour Detection / 阈值分割与轮廓检测

## Overview / 概述

This experiment covers image thresholding (segmentation) techniques and morphological contour detection. It includes both automatic threshold selection methods and classic morphological operations for boundary extraction.

本实验涵盖图像阈值分割技术和形态学轮廓检测，包括自动阈值选择方法和经典的形态学边界提取操作。

---

## Programs / 程序列表

### 1. OTSU Threshold / OTSU阈值 (`OTSUThreshold/`)

**Language:** MATLAB

**Description / 描述:**
Implements **OTSU's method** (Otsu's method, 1979) for automatic threshold selection. The algorithm:

实现**OTSU法**（大津法，1979年）进行自动阈值选择。算法步骤：

1. **Compute histogram / 计算直方图** — Count pixels at each gray level
2. **Compute probability distribution / 计算概率分布** — Normalize histogram by total pixels
3. **Iterate all thresholds (0-255):**
   - Split pixels into two classes (background/foreground)
   - Compute class weights (w1, w2) and class means (u1, u2)
   - Compute **between-class variance**: `σ_B² = w1 × w2 × (u1 - u2)²`
4. **Select optimal threshold** — The threshold that maximizes σ_B²

**Why OTSU works:** Maximum between-class variance means the two classes (foreground/background) are most separable.

**OTSU原理：** 最大类间方差意味着两个类别（前景/背景）最易分离。

**Files / 文件:**
- `main.m` — Main script
- `MyOTSU.m` — Core OTSU algorithm implementation

---

### 2. Iterative Optimal Threshold / 迭代最优阈值 (`IterativeOptimalThreshold/`)

**Language:** MATLAB

**Description / 描述:**
Implements **iterative threshold selection** — an alternative automatic thresholding method. The algorithm:

实现**迭代阈值选择**——另一种自动阈值方法。算法步骤：

1. **Initialize threshold / 初始化阈值** — T = (min + max) / 2 of image gray range
2. **Iterate until convergence:**
   - Split pixels into G1 (≥ T) and G2 (< T)
   - Compute mean of each group: u1 = mean(G1), u2 = mean(G2)
   - Update threshold: T_new = (u1 + u2) / 2
   - If |T_new - T| < ε (0.5), stop
3. **Apply final threshold** for binary segmentation

**Convergence guarantee:** The threshold always converges because the means u1 and u2 stabilize.

**收敛保证：** 由于均值u1和u2趋于稳定，阈值总会收敛。

**Files / 文件:**
- `main.m` — Main script with convergence plot
- `MyOptimalThreshold.m` — Core iterative algorithm

---

### 3. P-Percent Threshold / P率阈值化 (`PThreshold.cpp`)

**Language:** C++ / OpenCV

**Description / 描述:**
Implements **P-percent thresholding** — an adaptive threshold method where the threshold T is chosen such that exactly p% of pixels are darker (or lighter) than T.

实现**P率阈值化**——一种自适应阈值方法，选择阈值T使得恰好p%的像素比T更暗（或更亮）。

1. **Compute histogram / 计算直方图**
2. **Accumulate from gray level 0 upward / 从灰度级0开始累加**
3. **Find threshold T / 找到阈值T** — where cumulative count ≥ p × totalPixels
4. **Binarize / 二值化** — Pixels ≥ T → white (foreground), pixels < T → black (background)

Default p = 0.5 (median split), adjustable to other values.

默认p = 0.5（中位数分割），可调整为其他值。

**Files / 文件:**
- `PThreshold.cpp` — Complete P-percent thresholding implementation

---

### 4. Inner Contour Detection / 内边界跟踪 (`InnerContour.cpp`)

**Language:** C++ / OpenCV

**Description / 描述:**
Detects object contours using **morphological erosion** (morphological differencing). The algorithm:

使用**形态学腐蚀**（形态学差分）检测物体轮廓。算法步骤：

1. **Erosion / 腐蚀** — For each pixel, check if all 9 pixels in the 3×3 neighborhood are foreground (255). If yes, keep as foreground; otherwise, set to background.
2. **Contour extraction / 轮廓提取** — `Contour = Original - Eroded`

**Key insight:** Foreground interior pixels remain after erosion (they have 8 foreground neighbors), but boundary pixels become background (missing at least one foreground neighbor). The difference reveals both outer and inner contours.

**核心原理：** 前景内部像素在腐蚀后保留（有8个前景邻居），但边界像素变为背景（缺少至少一个前景邻居）。差值同时显示外边界和内边界。

**Files / 文件:**
- `InnerContour.cpp` — Complete contour detection implementation

---

## Notes / 注意事项

- MATLAB programs use `mytest.jpg` as input; C++ programs use `testimg.jpg`
- MATLAB程序使用`mytest.jpg`作为输入；C++程序使用`testimg.jpg`
- OTSU and Iterative methods produce similar results; OTSU is more rigorous mathematically
- OTSU法和迭代法结果相似；OTSU在数学上更严谨
- Inner contour detection uses OTSU thresholding as preprocessing step
- 内边界检测使用OTSU阈值法作为预处理步骤
