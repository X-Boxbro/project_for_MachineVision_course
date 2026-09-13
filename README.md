# 机器视觉课程项目

English version: [README_en.md](README_en.md)

---

## 项目简介

本仓库包含**机器视觉**课程的全部实验和课程设计项目。项目涵盖基本图像处理技术、形态学操作、阈值分割、边缘检测、滤波以及基于深度学习的手势识别。

---

## 目录结构

```
project_repo/
├── CourseDesign/          # 课程设计 - 手势识别
│   ├── Part1_SVM_HandPose/        # 第一部分：基于SVM的手姿态识别
│   └── Part2_DeepLearning_HandGesture/  # 第二部分：ConvNeXt深度学习手势识别
├── Experiment1/           # 实验1 - 基础图像处理 (MATLAB + C++/OpenCV)
├── Experiment2/           # 实验2 - 滤波与边缘检测 (MATLAB + C++/OpenCV)
└── Experiment3/           # 实验3 - 阈值分割与轮廓检测 (MATLAB + C++/OpenCV)
```

---

## 实验说明

### 实验1 - 基础图像处理

涵盖基本图像处理操作，包括距离变换、直方图计算与显示、卷积（空间域和基于FFT）以及积分图像计算。

- **距离变换** — 使用Chamfer距离变换算法 (MATLAB)
- **直方图显示** — 手动计算并可视化灰度直方图 (C++/OpenCV)
- **卷积** — 高斯卷积的直接空间计算与FFT方法 (C++/OpenCV)
- **积分图像** — 使用递归公式计算求和面积表 (MATLAB)

> 详细说明：[Experiment1/README.md](Experiment1/README.md)

### 实验2 - 滤波与边缘检测

涵盖中值滤波、Sobel边缘检测、直方图均衡化和线性（均值）滤波操作。

- **中值滤波** — 3×3中值滤波，复制边界填充 (MATLAB)
- **边缘检测** — Sobel算子边缘检测 (C++/OpenCV)
- **直方图均衡化** — 对比度增强 (C++/OpenCV)
- **线性滤波** — 3×3盒滤波器/均值滤波 (C++/OpenCV)

> 详细说明：[Experiment2/README.md](Experiment2/README.md)

### 实验3 - 阈值分割与轮廓检测

涵盖OTSU阈值法、迭代最优阈值选择、P率阈值化以及基于形态学操作的内边界检测。

- **OTSU阈值** — OTSU自动阈值法 (MATLAB)
- **迭代最优阈值** — 迭代阈值收敛算法 (MATLAB)
- **P率阈值化** — P率自适应阈值 (C++/OpenCV)
- **内边界跟踪** — 形态学腐蚀检测轮廓 (C++/OpenCV)

> 详细说明：[Experiment3/README.md](Experiment3/README.md)

### 课程设计 - 手势识别

综合性项目：第一部分使用SVM结合HOG特征进行手姿态分类（4类）；第二部分使用微调ConvNeXt-Tiny深度学习模型进行6类手势识别。

- **第一部分** — SVM + HOG特征手姿态识别 (C++/OpenCV)
- **第二部分** — ConvNeXt-Tiny深度学习手势识别 (Python/PyTorch)

> 详细说明：[CourseDesign/README.md](CourseDesign/README.md)

---

## 技术栈

| 组件 | 语言 | 库/工具 |
|------|------|---------|
| 实验1-3 (MATLAB部分) | MATLAB | Image Processing Toolbox |
| 实验1-3 (C++部分) | C++ | OpenCV 4.x |
| 课程设计第一部分 | C++ | OpenCV 4.x, SVM |
| 课程设计第二部分 | Python 3 | PyTorch, torchvision, PIL |

---

## 学生信息

- **姓名:** 谢刚
- **学号:** 2411628
- **课程:** 机器视觉
- **学期:** 大二下

