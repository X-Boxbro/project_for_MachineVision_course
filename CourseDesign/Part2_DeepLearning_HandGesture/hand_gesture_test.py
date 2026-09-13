"""
================================================================================
手势识别测试脚本 —— 深度学习 Part2
================================================================================

【模型信息】
- 网络架构: ConvNeXt-Tiny (基于 ImageNet-1K 预训练权重微调)
- 分类数量: 6 类 (A、B、C、Five、Point、V)
- 验证集准确率: 100%
- 测试集准确率: 99.72%

【使用方法】
1. 将数据集解压到本脚本同一目录（需要torch环境，具体请查看***requirements.txt***文件）
2. 修改下方 DATA_DIR 为你的数据集路径
3. 运行: python hand_gesture_test.py

【数据集路径配置】请修改下方 DATA_DIR 变量
"""
# ============================================================================
# >>>>>>>>>  请在这里修改数据集路径  <<<<<<<<<<
# ============================================================================
# 方式一：Hard 数据集（6类）
# DATA_DIR = "./Hand_Posture_Hard_Stu"

DATA_DIR = "E:/Hand_Posture_Easy_Stu"

# 方式三：自定义路径
# DATA_DIR = "./your_test_images"
# ============================================================================

import os
import sys
from pathlib import Path
from glob import glob

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


def build_model(num_classes: int = 6):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        torch.tensor([1.0]).cuda()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", ["A", "B", "C", "Five", "Point", "V"])
    num_classes = checkpoint.get("num_classes", 6)

    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[模型加载成功] 类别: {class_names}")
    print(f"[模型加载成功] 验证集准确率: {checkpoint.get('val_acc', 'N/A')}%")
    return model, class_names


def predict_image(model: nn.Module, image_path: str, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        conf, pred_idx = probs.max(1)
    return pred_idx.item(), conf.item()


def find_images_recursive(folder: str) -> list[str]:
    patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    images = []
    for pattern in patterns:
        images.extend(glob(os.path.join(folder, "**", pattern), recursive=True))
    return sorted(images)


def main():
    model_path = "best_model.pth"

    device = get_device()
    print("=" * 60)
    print("手势识别测试 —— ConvNeXt-Tiny 深度学习模型")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"数据集: {DATA_DIR}")
    print()

    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        sys.exit(1)

    model, class_names = load_model(model_path, device)
    print()

    image_paths = find_images_recursive(DATA_DIR)
    if not image_paths:
        print(f"[错误] 未找到图片: {DATA_DIR}")
        sys.exit(1)

    print(f"找到 {len(image_paths)} 张图片，开始识别...")
    print("-" * 60)

    correct = 0
    results = []
    per_class_correct = {cn: 0 for cn in class_names}
    per_class_total = {cn: 0 for cn in class_names}

    for img_path in image_paths:
        pred_idx, confidence = predict_image(model, img_path, device)
        pred_name = class_names[pred_idx]
        img_name = os.path.basename(img_path)
        results.append((img_name, pred_name, confidence, img_path))

        true_label = None
        for part in Path(img_path).parts:
            for cn in class_names:
                if part.startswith(cn):
                    true_label = cn
                    break
            if true_label:
                break
        if not true_label:
            for cn in class_names:
                if img_name.startswith(cn):
                    true_label = cn
                    break

        print(f"{img_name:40s} -> {pred_name:8s} ({confidence:.4f})")

        if true_label:
            per_class_total[true_label] += 1
            if true_label == pred_name:
                correct += 1
                per_class_correct[true_label] += 1

    print("-" * 60)
    print(f"\n识别完成！共 {len(image_paths)} 张图片")

    if per_class_total:
        print("\n各类别准确率:")
        print("-" * 40)
        for cn in class_names:
            if per_class_total[cn] > 0:
                acc = 100.0 * per_class_correct[cn] / per_class_total[cn]
                print(f"  {cn:8s}: {per_class_correct[cn]:4d}/{per_class_total[cn]:4d} = {acc:.2f}%")
        total = sum(per_class_total.values())
        print("-" * 40)
        print(f"  总体准确率: {correct}/{total} = {100.0*correct/total:.2f}%")

    with open("prediction_results.txt", "w", encoding="utf-8") as f:
        f.write("filename,predicted_class,confidence,full_path\n")
        for img_name, pred_name, conf, img_path in results:
            f.write(f"{img_name},{pred_name},{conf:.6f},{img_path}\n")
    print(f"\n结果已保存: prediction_results.txt")


if __name__ == "__main__":
    main()
