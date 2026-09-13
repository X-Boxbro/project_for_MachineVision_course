"""
手势识别训练脚本 —— ConvNeXt-Tiny 微调
识别类别：A、B、C、Five、Point、V（共6类）
使用 ImageNet 预训练权重进行迁移学习
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from dataset import (
    DATA_ROOT,
    CLASS_NAMES,
    NUM_CLASSES,
    build_dataloaders,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    构建 ConvNeXt-Tiny 模型，加载 ImageNet 预训练权重，
    替换最后分类头为6类输出
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)

    # ConvNeXt 的分类头是一个 Linear 层，名为 'classifier'
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def get_device() -> torch.device:
    """优先使用 CUDA，否则 fallback 到 CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # CUDA 可能存在但 is_available 返回 False，手动尝试
    try:
        torch.tensor([1.0]).cuda()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="手势识别 ConvNeXt-Tiny 训练")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"设备: {device}")
    print(f"数据目录: {args.data_root}")
    print(f"训练轮次: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"图像尺寸: {args.img_size}")
    print(f"=" * 60)

    # 数据加载
    train_loader, val_loader = build_dataloaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"类别: {CLASS_NAMES}")
    print()

    # 模型
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(device)
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    print("开始训练...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        epoch_str = f"[{epoch}/{args.epochs}]"

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - start_time

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        }
        history.append(record)

        print(
            f"{epoch_str} "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {record['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_path = save_dir / "best_model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "num_classes": NUM_CLASSES,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  >> 保存最优模型: {best_path} (Val Acc: {val_acc:.2f}%)")

    print("-" * 60)
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # 保存完整训练历史
    history_path = save_dir / "training_history.txt"
    with open(history_path, "w") as f:
        f.write(
            "epoch,train_loss,train_acc,val_loss,val_acc,lr,time(s)\n"
        )
        for r in history:
            f.write(
                f"{r['epoch']},{r['train_loss']:.6f},{r['train_acc']:.4f},"
                f"{r['val_loss']:.6f},{r['val_acc']:.4f},"
                f"{r['lr']:.6e},{r['time']:.2f}\n"
            )
    print(f"训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
