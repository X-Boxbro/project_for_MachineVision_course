# Experiment 2 - Filtering & Edge Detection

中文版：[README.md](README.md)

---

## Overview

This experiment covers spatial domain filtering and edge detection techniques. It includes both linear and non-linear filters, as well as a classic differential-based edge detector (Sobel operator).

---

## Programs

### 1. Median Filter (`MedianFilter/`)

**Language:** MATLAB

**Description:**
Implements a **3×3 median filter** for noise removal. Unlike linear filters (mean/box filter), the median filter is a **nonlinear** operation that:

1. Extracts the 3×3 neighborhood around each pixel
2. Sorts the 9 pixel values in ascending order
3. Takes the median (5th element) as the output

The median filter is particularly effective at removing **salt-and-pepper noise** while preserving edges better than mean filtering. **Replicate boundary padding** is used.

**Files:**
- `main.m` — Main script with environment setup, image loading, and result display
- `MyMedfilt2.m` — Core median filter implementation

---

### 2. Edge Detection (`EdgeDetection.cpp`)

**Language:** C++ / OpenCV

**Description:**
Implements the **Sobel operator** for edge detection. The Sobel operator uses two 3×3 convolution kernels:

- **Gx kernel (horizontal gradient):** Detects vertical edges (left-to-right intensity changes)
- **Gy kernel (vertical gradient):** Detects horizontal edges (top-to-bottom intensity changes)

The final edge magnitude is computed as: `G = sqrt(Gx² + Gy²)`

Boundary handling uses **replicate** mode (edge pixels are clamped to boundary values).

**Files:**
- `EdgeDetection.cpp` — Complete Sobel edge detection implementation

---

### 3. Histogram Equalization (`HistogramEqualization.cpp`)

**Language:** C++ / OpenCV

**Description:**
Implements **histogram equalization** for automatic contrast enhancement. The algorithm:

1. **Compute histogram** — Count occurrences of each gray level (0-255)
2. **Compute cumulative histogram (CDF)** — Cumulative sum of histogram
3. **Build lookup table (LUT)** — `map[k] = round(255 * CDF[k] / totalPixels)`
4. **Apply mapping** — Replace each pixel value using the LUT

Result: The output image has a more uniform histogram distribution, enhancing overall contrast.

**Files:**
- `HistogramEqualization.cpp` — Complete histogram equalization implementation

---

### 4. Linear Filter (`LinearFilter.cpp`)

**Language:** C++ / OpenCV

**Description:**
Implements a **3×3 box filter (mean filter)** — the simplest linear filter. The kernel uses equal weights:

```
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

Each pixel is replaced by the arithmetic mean of its 3×3 neighborhood. This smooths the image and reduces noise, but also causes blurring (loss of detail).

**Files:**
- `LinearFilter.cpp` — Complete box filter implementation

---

## Notes

- All programs read `testimg.jpg` as input (replaceable)
- Boundary handling: replicate mode for both C++ and MATLAB implementations
