# Experiment 3 - Thresholding & Contour Detection

中文版：[README.md](README.md)

---

## Overview

This experiment covers image thresholding (segmentation) techniques and morphological contour detection. It includes both automatic threshold selection methods and classic morphological operations for boundary extraction.

---

## Programs

### 1. OTSU Threshold (`OTSUThreshold/`)

**Language:** MATLAB

**Description:**
Implements **OTSU's method** (Otsu's method, 1979) for automatic threshold selection. The algorithm:

1. **Compute histogram** — Count pixels at each gray level
2. **Compute probability distribution** — Normalize histogram by total pixels
3. **Iterate all thresholds (0-255):**
   - Split pixels into two classes (background/foreground)
   - Compute class weights (w1, w2) and class means (u1, u2)
   - Compute **between-class variance**: `σ_B² = w1 × w2 × (u1 - u2)²`
4. **Select optimal threshold** — The threshold that maximizes σ_B²

**Why OTSU works:** Maximum between-class variance means the two classes (foreground/background) are most separable.

**Files:**
- `main.m` — Main script
- `MyOTSU.m` — Core OTSU algorithm implementation

---

### 2. Iterative Optimal Threshold (`IterativeOptimalThreshold/`)

**Language:** MATLAB

**Description:**
Implements **iterative threshold selection** — an alternative automatic thresholding method. The algorithm:

1. **Initialize threshold** — T = (min + max) / 2 of image gray range
2. **Iterate until convergence:**
   - Split pixels into G1 (≥ T) and G2 (< T)
   - Compute mean of each group: u1 = mean(G1), u2 = mean(G2)
   - Update threshold: T_new = (u1 + u2) / 2
   - If |T_new - T| < ε (0.5), stop
3. **Apply final threshold** for binary segmentation

**Convergence guarantee:** The threshold always converges because the means u1 and u2 stabilize.

**Files:**
- `main.m` — Main script with convergence plot
- `MyOptimalThreshold.m` — Core iterative algorithm

---

### 3. P-Percent Threshold (`PThreshold.cpp`)

**Language:** C++ / OpenCV

**Description:**
Implements **P-percent thresholding** — an adaptive threshold method where the threshold T is chosen such that exactly p% of pixels are darker (or lighter) than T.

1. **Compute histogram**
2. **Accumulate from gray level 0 upward**
3. **Find threshold T** — where cumulative count ≥ p × totalPixels
4. **Binarize** — Pixels ≥ T → white (foreground), pixels < T → black (background)

Default p = 0.5 (median split), adjustable to other values.

**Files:**
- `PThreshold.cpp` — Complete P-percent thresholding implementation

---

### 4. Inner Contour Detection (`InnerContour.cpp`)

**Language:** C++ / OpenCV

**Description:**
Detects object contours using **morphological erosion** (morphological differencing). The algorithm:

1. **Erosion** — For each pixel, check if all 9 pixels in the 3×3 neighborhood are foreground (255). If yes, keep as foreground; otherwise, set to background.
2. **Contour extraction** — `Contour = Original - Eroded`

**Key insight:** Foreground interior pixels remain after erosion (they have 8 foreground neighbors), but boundary pixels become background (missing at least one foreground neighbor). The difference reveals both outer and inner contours.

**Files:**
- `InnerContour.cpp` — Complete contour detection implementation

---

## Notes

- MATLAB programs use `mytest.jpg` as input; C++ programs use `testimg.jpg`
- OTSU and Iterative methods produce similar results; OTSU is more rigorous mathematically
- Inner contour detection uses OTSU thresholding as preprocessing step
