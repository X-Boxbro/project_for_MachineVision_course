# 课程设计 - 手势识别

English version: [README_en.md](README_en.md)

---

## 概述

本课程设计是机器视觉课程的综合性项目，分为两个部分：

- **第一部分：** 使用SVM结合HOG特征的传统机器学习方法进行手姿态分类（4类）
- **第二部分：** 使用微调ConvNeXt-Tiny深度学习模型进行手势识别（6类）

---

## 第一部分 - SVM手姿态识别 (C++/OpenCV)

**作者:** 谢刚 | **学号:** 2411628

### 描述

基于**支持向量机(SVM)**结合**方向梯度直方图(HOG)**特征和**PCA**降维的手姿态识别系统。

### 主要特点

| 特性 | 详情 |
|------|------|
| **算法** | SVM (RBF核) + PCA |
| **特征** | HOG (128×128窗口, 16×16块, 8×8单元格, 9方向) |
| **类别** | A, C, Five, V (4类) |
| **预处理** | Gamma校正 → CLAHE → 高斯模糊 → OTSU阈值化 → 形态学闭运算 |

### 文件结构

```
Part1_SVM_HandPose/
├── hand_pose_inference.cpp      # 主推理程序
├── hand_pose_svm_model.yml      # SVM模型文件
├── hand_pose_svm_model_params.yml  # PCA参数
└── requirements.txt             # 环境配置说明
```

### 环境要求

- OpenCV ≥ 4.0 (core, imgproc, imgcodecs, ml)
- C++11或更高
- 兼容Windows (MinGW/MSVC)和Linux

### 使用方法

1. 使用OpenCV编译：`g++ -o hand_pose_inference hand_pose_inference.cpp ...`
2. 将模型文件放在同一目录
3. 在源代码中更新`TEST_FOLDER`路径
4. 运行：`./hand_pose_inference`

---

## 第二部分 - ConvNeXt-Tiny深度学习 (Python/PyTorch)

### 描述

使用**ConvNeXt-Tiny**架构、基于ImageNet-1K预训练权重微调的深度学习手势识别系统。

### 主要特点

| 特性 | 详情 |
|------|------|
| **架构** | ConvNeXt-Tiny |
| **类别** | A, B, C, Five, Point, V (6类) |
| **预处理** | Resize 224×224 → 归一化 (ImageNet统计值) |
| **验证集准确率** | 100% |
| **测试集准确率** | 99.72% |

### 文件结构

```
Part2_DeepLearning_HandGesture/
├── hand_gesture_test.py         # 主测试脚本
├── best_model.pth               # 训练好的模型权重 (111MB)
└── requirements.txt             # Python依赖
```

### 环境要求

```
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
numpy>=1.20.0
```

### 使用方法

1. 安装依赖：`pip install -r requirements.txt`
2. 在脚本中修改`DATA_DIR`为你的数据集路径
3. 运行：`python hand_gesture_test.py`

---

## 注意事项

- 第一部分模型文件(`*_svm_model.yml`, `*_svm_model_params.yml`)因体积较大已在.gitignore中
- 第二部分`best_model.pth`(111MB)超过GitHub 100MB限制，已在.gitignore中
- 这些文件需要单独下载或生成
