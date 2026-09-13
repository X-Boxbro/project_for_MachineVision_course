"""
手势识别数据集加载与增强
识别类别：A、B、C、Five、Point、V（共6类）
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


# 数据集根目录
DATA_ROOT = Path(__file__).parent / "data" / "Hand_Posture_Hard_Stu"

# 类别名称（按文件夹顺序）
CLASS_NAMES = ["A", "B", "C", "Five", "Point", "V"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class HandPostureDataset(Dataset):
    """手势姿态数据集，从按类别分类的文件夹中加载图像"""

    def __init__(
        self,
        root: Path,
        class_names: List[str],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        self.root = Path(root)
        self.class_names = class_names
        self.transform = transform
        self.extensions = extensions

        self.samples = []  # List of (image_path, class_index)
        self._load_samples()

    def _load_samples(self):
        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            class_idx = CLASS_TO_IDX[class_name]
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in self.extensions:
                    self.samples.append((str(img_file), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """训练集数据增强"""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    """验证/测试集预处理（无数据增强）"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """按类别分层采样，划分为训练集和验证集"""
    random.seed(seed)
    torch.manual_seed(seed)

    # 按类别分组
    class_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    train_indices, val_indices = [], []
    for label, indices in class_indices.items():
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_ratio))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_dataloaders(
    data_root: Path = DATA_ROOT,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """构建训练/验证 DataLoader"""
    train_dataset = HandPostureDataset(
        root=data_root,
        class_names=CLASS_NAMES,
        transform=get_train_transform(img_size),
    )
    val_dataset = HandPostureDataset(
        root=data_root,
        class_names=CLASS_NAMES,
        transform=get_val_transform(img_size),
    )

    train_subset, val_subset = split_dataset(train_dataset, val_ratio=val_ratio)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # 快速验证数据集加载
    train_loader, val_loader = build_dataloaders(batch_size=8, num_workers=0)
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"类别: {CLASS_NAMES}")

    # 可视化一个批次
    images, labels = next(iter(train_loader))
    print(f"批次形状: {images.shape}, 标签形状: {labels.shape}")
