# Experiment 1 - Basic Image Processing

中文版：[README.md](README.md)

---

## Overview

This experiment covers four fundamental image processing operations, implemented in both MATLAB and C++/OpenCV. These operations form the foundation of many computer vision algorithms.

---

## Programs

### 1. Distance Transform (`DistanceTransform/`)

**Language:** MATLAB

**Description:**
The distance transform computes the distance from each pixel to the nearest foreground (non-zero) pixel in a binary image. This implementation uses the **chamfer distance transform** algorithm, which performs two passes over the image:

- **First pass (top-left to bottom-right):** Checks top and left neighbors
- **Second pass (bottom-right to top-left):** Checks bottom and right neighbors, updates minimum

**Files:**
- `main.m` — Main script, reads image, converts to binary, calls MyDisTrans
- `MyDisTrans.m` — Core algorithm implementation

---

### 2. Histogram Display (`HistogramDisplay.cpp`)

**Language:** C++ / OpenCV

**Description:**
Manually computes and visualizes the grayscale histogram of an image. Instead of using OpenCV's built-in `calcHist` and `normalize` functions, this program:

1. Iterates through all pixels to count occurrences of each gray level (0-255)
2. Normalizes the histogram to fit the display canvas
3. Draws the histogram as a bar chart using `rectangle()`

**Files:**
- `HistogramDisplay.cpp` — Complete implementation

---

### 3. Convolution (`Convolution.cpp`)

**Language:** C++ / OpenCV

**Description:**
Implements Gaussian convolution using two different methods:

1. **Direct spatial convolution** — Applies a 128×128 Gaussian kernel using nested loops (4-layer) with border skipping (BORDER_CONSTANT)
2. **FFT-based convolution** — Pads image and kernel to optimal DFT size, performs DFT, multiplies in frequency domain, then inverse DFT to obtain result

The Gaussian kernel is generated with σ=20.0 and normalized so that all weights sum to 1.

**Files:**
- `Convolution.cpp` — Both implementations (spatial and FFT-based)

---

### 4. Integral Image (`IntegralImage/`)

**Language:** MATLAB

**Description:**
Computes the **summed-area table** (integral image) of a grayscale image. The integral image at position (i,j) contains the sum of all pixel values above and to the left:

```
SA(i,j) = img(i,j) + SA(i-1,j) + SA(i,j-1) - SA(i-1,j-1)
```

This recursive formula allows O(1) computation of any rectangular region sum after O(M×N) preprocessing.

**Files:**
- `main.m` — Main script
- `MyInteImg.m` — Core algorithm implementation

---

## Notes

- Input images can be freely replaced
- In MATLAB, computed matrix data cannot be directly displayed as an image — type conversion (e.g., `uint8()`) may be required
