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
├── hand_pose_train.cpp          # 训练代码 (SVM+HOG特征提取与训练)
├── hand_pose_inference.cpp      # 推理/测试代码
├── hand_pose_svm_model.yml      # 训练好的SVM模型
├── hand_pose_svm_model_params.yml  # PCA归一化参数
└── requirements.txt             # 环境配置说明
```

### 训练代码说明

`hand_pose_train.cpp` 是第一部分的核心训练代码，包含：
- HOG + LBP + Hu矩 + 轮廓特征融合
- PCA降维
- K折交叉验证自动选择最优SVM模型
- 数据增强（亮度变化、水平翻转、旋转、缩放、高斯噪声）

> 注：当前文件中代码以注释形式保留（因编译环境依赖），使用时取消注释并配置数据集路径即可编译运行。

### 环境要求

- OpenCV ≥ 4.0 (core, imgproc, imgcodecs, ml)
- C++11或更高
- 兼容Windows (MinGW/MSVC)和Linux

### 使用方法

1. 使用OpenCV编译：`g++ -o hand_pose_train hand_pose_train.cpp ...`
2. 将模型文件放在同一目录
3. 在源代码中更新`TRAIN_ROOT`数据集路径
4. 运行：`./hand_pose_train`

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
├── train.py                     # 训练代码 (ConvNeXt-Tiny微调)
├── dataset.py                   # 数据集加载与增强
├── hand_gesture_test.py         # 测试/推理代码
├── training_history.txt         # 训练历史记录 (30 epochs)
├── best_model.pth               # 训练好的模型权重 (111MB)
└── requirements.txt             # Python依赖
```

### 训练代码说明

`train.py` 是第二部分的核心训练代码，包含：
- ConvNeXt-Tiny 模型构建 (ImageNet预训练权重)
- AdamW优化器 + 余弦退火学习率调度
- 训练/验证分离 (80/20分层采样)
- 自动保存最优模型
- 训练历史记录保存

训练数据集结构：
```
Hand_Posture_Hard_Stu/
├── A/         # A类手势图像
├── B/         # B类手势图像
├── C/         # C类手势图像
├── Five/      # Five类手势图像
├── Point/     # Point类手势图像
└── V/         # V类手势图像
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
2. 训练：`python train.py --data_root <数据集路径>`
3. 测试：`python hand_gesture_test.py`

---

## 注意事项

- 第一部分模型文件(`*_svm_model.yml`, `*_svm_model_params.yml`)因体积较大已在.gitignore中
- 第二部分`best_model.pth`(111MB)超过GitHub 100MB限制，已在.gitignore中
- 这些文件需要单独下载或生成
