"""
Hybrid Ensemble Training Script for Cucumber Disease Classification

Sections:
- 1) Configuration & Dependencies
- 2) Reproducibility & Utilities
- 3) Dataset Discovery & Stratified Train/Val/Test Split
- 4) Preprocessing & Augmentations
- 5) Dataset and DataLoaders (with class balancing)
- 6) Model Factory (timm backbones) and Two-Stage Fine-Tuning
- 7) Training Loop with Anti-Overfitting (Mixup/CutMix, Label Smoothing, Weight Decay, AMP)
- 8) Early Stopping, Scheduler, and Logging
- 9) Evaluation: Confusion Matrix, Accuracy, Precision, Recall, F1, ROC; Plots
- 10) Ensembling (Probability Averaging) and Test-Time Augmentation (optional)
- 11) Main: End-to-end run

Usage:
- Set CFG.DATASET_DIR to your dataset root. Expected structure:
    DATASET_DIR/
      class_0/*.jpg|png
      class_1/*.jpg|png
      ...
- Or if you already have split folders:
    DATASET_DIR/train/<class>/*.jpg
    DATASET_DIR/val/<class>/*.jpg   # optional, will be created if absent
    DATASET_DIR/test/<class>/*.jpg  # optional, will be created if absent

Dependencies (install in Kaggle/Colab/Local beforehand):
    pip install timm==1.0.9 albumentations==1.4.6 torchmetrics==1.4.2 seaborn==0.13.2
"""

from __future__ import annotations

import os
import math
import time
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings("ignore")


# =============================
# 1) Configuration & Dependencies
# =============================
class CFG:
    # Update this to your dataset root path
    DATASET_DIR: str = os.environ.get("CUCUMBER_DATASET_DIR", "/kaggle/input/cucumber-disease-dataset")

    # General
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "./outputs")
    RANDOM_SEED: int = 42
    NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", 2))
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Images
    IMAGE_SIZE: int = int(os.environ.get("IMAGE_SIZE", 224))  # try 384 if GPU allows

    # Split ratios used if split folders are not provided
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1

    # Training
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", 32))
    EPOCHS_STAGE1: int = int(os.environ.get("EPOCHS_STAGE1", 3))      # frozen backbone
    EPOCHS_STAGE2: int = int(os.environ.get("EPOCHS_STAGE2", 20))     # full fine-tune
    LR: float = float(os.environ.get("LR", 3e-4))
    WEIGHT_DECAY: float = float(os.environ.get("WEIGHT_DECAY", 1e-4))
    LABEL_SMOOTHING: float = float(os.environ.get("LABEL_SMOOTHING", 0.1))
    DROPOUT: float = float(os.environ.get("DROPOUT", 0.2))
    USE_AMP: bool = os.environ.get("USE_AMP", "1") == "1"

    # Mixup/CutMix strategy
    MIXUP_ALPHA: float = float(os.environ.get("MIXUP_ALPHA", 0.2))
    CUTMIX_ALPHA: float = float(os.environ.get("CUTMIX_ALPHA", 1.0))
    MIXUP_EPOCHS: int = int(os.environ.get("MIXUP_EPOCHS", 12))  # disable mixup in final epochs for sharper decision boundaries

    # Regularization & scheduling
    EARLY_STOPPING_PATIENCE: int = int(os.environ.get("EARLY_STOPPING_PATIENCE", 6))
    SCHED_T0: int = int(os.environ.get("SCHED_T0", 6))
    SCHED_ETA_MIN_SCALE: float = float(os.environ.get("SCHED_ETA_MIN_SCALE", 1e-2))

    # TTA and Ensembling
    TTA: bool = os.environ.get("TTA", "0") == "1"
    MODEL_NAMES: List[str] = [
        # Complementary strong backbones; adjust per compute
        # You may switch to base sizes if GPU is limited
        "tf_efficientnetv2_s.in21k_ft_in1k",
        "convnext_small_in22k",
        "swin_small_patch4_window7_224",
    ]


# =============================
# 2) Reproducibility & Utilities
# =============================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_device() -> torch.device:
    return torch.device(CFG.DEVICE)


set_seed(CFG.RANDOM_SEED)
ensure_dir(CFG.OUTPUT_DIR)
device = get_device()
print(f"Using device: {device}")


# =============================
# 3) Dataset Discovery & Stratified Train/Val/Test Split
# =============================
def discover_images_from_dir(root: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    class_names = sorted([p.name for p in class_dirs])
    assert len(class_names) >= 2, f"Need at least 2 classes under {root}"

    rows: List[Tuple[str, str]] = []
    for cls in class_names:
        for p in (root / cls).rglob("*"):
            if p.is_file() and p.suffix.lower() in image_extensions:
                rows.append((str(p), cls))

    df = pd.DataFrame(rows, columns=["filepath", "label"])
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    df["label_id"] = df["label"].map(label_to_id)
    return df, class_names, label_to_id


def has_split_folders(dataset_dir: Path) -> bool:
    return (dataset_dir / "train").exists()


def load_or_split_dataset(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, int]]:
    root = Path(dataset_dir)
    assert root.exists(), f"Dataset dir not found: {root}"

    if has_split_folders(root):
        print("Detected split folders under dataset root: using train/val/test if present")

        # train
        train_df, class_names, label_to_id = discover_images_from_dir(root / "train")

        # val (optional) else create from train
        val_root = root / "val"
        if val_root.exists():
            val_df, _, _ = discover_images_from_dir(val_root)
        else:
            print("val/ not found: creating a stratified split from train/")
            train_df, val_df = stratified_split(train_df, val_size=CFG.VAL_RATIO)

        # test (optional) else create from train leftovers
        test_root = root / "test"
        if test_root.exists():
            test_df, _, _ = discover_images_from_dir(test_root)
        else:
            print("test/ not found: creating a stratified split from train/ leftovers")
            # Split from train_df again into train/test with small ratio
            train_df, test_df = stratified_split(train_df, val_size=CFG.TEST_RATIO)

        return train_df, val_df, test_df, class_names, label_to_id

    # No pre-split: scan and split into train/val/test
    print("No split folders found: performing stratified train/val/test split from a single directory")
    df_all, class_names, label_to_id = discover_images_from_dir(root)
    train_df, temp_df = stratified_split(df_all, val_size=(1.0 - CFG.TRAIN_RATIO))
    rel_val = CFG.VAL_RATIO / (CFG.VAL_RATIO + CFG.TEST_RATIO)
    val_df, test_df = stratified_split(temp_df, val_size=(1.0 - rel_val))
    return train_df, val_df, test_df, class_names, label_to_id


def stratified_split(df: pd.DataFrame, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=CFG.RANDOM_SEED)
    X = df["filepath"].values
    y = df["label_id"].values
    idx_train, idx_val = next(splitter.split(X, y))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_val].reset_index(drop=True)


train_df, val_df, test_df, CLASS_NAMES, LABEL_TO_ID = load_or_split_dataset(CFG.DATASET_DIR)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# =============================
# 4) Preprocessing & Augmentations
# =============================
def get_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.75, 1.0), ratio=(0.8, 1.25), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(scale=(0.9, 1.1), rotate=(-20, 20), shear=(-8, 8), p=0.7),
        A.RandomBrightnessContrast(p=0.6),
        A.HueSaturationValue(p=0.5),
        A.CoarseDropout(
            max_holes=8,
            max_height=int(image_size * 0.08),
            max_width=int(image_size * 0.08),
            min_holes=2,
            fill_value=0,
            p=0.5,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# =============================
# 5) Dataset and DataLoaders (with class balancing)
# =============================
class ImageFolderDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: A.Compose | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row.filepath).convert("RGB")
        image_np = np.array(image)
        if self.transforms is not None:
            image_t = self.transforms(image=image_np)["image"]
        else:
            image_t = ToTensorV2()(image=image_np)["image"]
        label = int(row.label_id)
        return image_t, label


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=labels.max() + 1)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)


def build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tfms = get_train_transforms(CFG.IMAGE_SIZE)
    eval_tfms = get_eval_transforms(CFG.IMAGE_SIZE)

    train_ds = ImageFolderDataset(train_df, transforms=train_tfms)
    val_ds = ImageFolderDataset(val_df, transforms=eval_tfms)
    test_ds = ImageFolderDataset(test_df, transforms=eval_tfms)

    sampler = create_weighted_sampler(train_df["label_id"].values)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, sampler=sampler, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df)


# =============================
# 6) Model Factory and Two-Stage Fine-Tuning
# =============================
def create_timm_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, drop_rate=dropout)
    return model


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_backbone(model: nn.Module) -> None:
    # Best effort freezing: keep classifier head trainable
    head_names = {"classifier", "head", "fc", "pre_logits", "head_drop"}
    for name, param in model.named_parameters():
        if any(hn in name for hn in head_names):
            param.requires_grad = True
        else:
            param.requires_grad = False


# =============================
# 7) Training Loop with Anti-Overfitting
# =============================
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn, scaler: torch.cuda.amp.GradScaler, mixup_fn: Mixup | None) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, labels)
        else:
            targets = labels

        optimizer.zero_grad(set_to_none=True)

        if CFG.USE_AMP:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        if mixup_fn is None:
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = (correct / total) if total > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item() * images.size(0)

        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    eval_loss = running_loss / len(loader.dataset)
    probs_np = np.concatenate(all_probs)
    preds_np = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    eval_acc = (preds_np == labels_np).mean()
    return eval_loss, eval_acc, probs_np, preds_np, labels_np


def get_scheduler(optimizer: torch.optim.Optimizer) -> CosineAnnealingWarmRestarts:
    eta_min = CFG.LR * CFG.SCHED_ETA_MIN_SCALE
    return CosineAnnealingWarmRestarts(optimizer, T_0=CFG.SCHED_T0, T_mult=1, eta_min=eta_min)


class EarlyStopping:
    def __init__(self, patience: int = 6, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improvement = (value - self.best) if self.mode == "max" else (self.best - value)
        if improvement > self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================
# 8) Training wrapper per model
# =============================
def train_model(model_name: str, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    print(f"\n=== Training {model_name} ===")
    model = create_timm_model(model_name, num_classes=NUM_CLASSES, dropout=CFG.DROPOUT).to(device)

    # Two-stage: freeze backbone first
    freeze_backbone(model)

    # Losses
    ce_loss = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    soft_ce_loss = SoftTargetCrossEntropy()

    mixup_fn = Mixup(
        mixup_alpha=CFG.MIXUP_ALPHA,
        cutmix_alpha=CFG.CUTMIX_ALPHA,
        label_smoothing=CFG.LABEL_SMOOTHING,
        num_classes=NUM_CLASSES,
    )

    # Stage 1 optimizer/scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.USE_AMP)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = None
    best_val_acc = -np.inf
    early_stopper = EarlyStopping(patience=CFG.EARLY_STOPPING_PATIENCE, mode="max")

    # Stage 1: train head only (frozen backbone)
    for epoch in range(1, CFG.EPOCHS_STAGE1 + 1):
        current_mixup = mixup_fn  # use mixup early
        loss_fn = soft_ce_loss

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, mixup_fn=current_mixup)
        val_loss, val_acc, val_probs, val_preds, val_labels = evaluate(model, val_loader, ce_loss)
        scheduler.step(epoch + val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Stage1 Ep{epoch:02d}/{CFG.EPOCHS_STAGE1} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, f"{CFG.OUTPUT_DIR}/best_{model_name.replace('/', '_')}.pth")

        if early_stopper.step(val_acc):
            print("Early stopping triggered in Stage 1")
            break

    # Stage 2: unfreeze and fine-tune fully
    set_requires_grad(model, True)
    optimizer = AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer)
    early_stopper = EarlyStopping(patience=CFG.EARLY_STOPPING_PATIENCE, mode="max")

    for epoch in range(1, CFG.EPOCHS_STAGE2 + 1):
        # Use mixup for first MIXUP_EPOCHS of stage2
        use_mixup = epoch <= CFG.MIXUP_EPOCHS
        current_mixup = mixup_fn if use_mixup else None
        loss_fn = soft_ce_loss if use_mixup else ce_loss

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, mixup_fn=current_mixup)
        val_loss, val_acc, val_probs, val_preds, val_labels = evaluate(model, val_loader, ce_loss)
        scheduler.step(epoch + val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Stage2 Ep{epoch:02d}/{CFG.EPOCHS_STAGE2} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, f"{CFG.OUTPUT_DIR}/best_{model_name.replace('/', '_')}.pth")

        if early_stopper.step(val_acc):
            print("Early stopping triggered in Stage 2")
            break

    # Load best state and final val pass
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    val_loss, val_acc, val_probs, val_preds, val_labels = evaluate(model, val_loader, ce_loss)

    # Save training curves
    plot_training_curves(history, f"{model_name} Training", f"{CFG.OUTPUT_DIR}/training_{model_name.replace('/', '_')}.png")

    metrics = {"best_val_acc": float(best_val_acc), "final_val_acc": float(val_acc)}
    return model, history, metrics, (val_probs, val_preds, val_labels)


# =============================
# 9) Metrics & Plots
# =============================
def plot_training_curves(history: Dict[str, List[float]], title: str, out_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, history["train_loss"], label="train_loss")
    ax[0].plot(epochs, history["val_loss"], label="val_loss")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(epochs, history["train_acc"], label="train_acc")
    ax[1].plot(epochs, history["val_acc"], label="val_acc")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix_and_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path_prefix: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(f"{out_path_prefix}_confusion_matrix.png")
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(f"{out_path_prefix}_classification_report.txt", "w") as f:
        f.write(report)
    print(report)


def plot_multiclass_roc(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], out_path: str) -> None:
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fpr: Dict[int, np.ndarray] = {}
    tpr: Dict[int, np.ndarray] = {}
    roc_auc: Dict[int, float] = {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(class_names):
        ax.plot(fpr[i], tpr[i], label=f"{cls} (AUC={roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (OvR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@torch.no_grad()
def predict_proba(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    model.eval()
    outputs = model(images)
    return F.softmax(outputs, dim=1)


@torch.no_grad()
def eval_with_tta(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        flips = [
            images,
            torch.flip(images, dims=[3]),
            torch.flip(images, dims=[2]),
            torch.flip(torch.flip(images, dims=[2]), dims=[3]),
        ]
        probs_sum = 0
        for img in flips:
            probs_sum += predict_proba(model, img)
        probs = (probs_sum / len(flips)).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(labels.numpy())
    probs_all_np = np.concatenate(probs_all)
    labels_all_np = np.concatenate(labels_all)
    preds_all_np = probs_all_np.argmax(axis=1)
    return probs_all_np, preds_all_np, labels_all_np


# =============================
# 10) Main: train all models, ensemble, and evaluate on test
# =============================
def main() -> None:
    ensure_dir(CFG.OUTPUT_DIR)

    models: List[Tuple[str, nn.Module]] = []
    histories: Dict[str, Dict[str, List[float]]] = {}
    single_val_probs: List[np.ndarray] = []
    single_val_preds: List[np.ndarray] = []
    single_test_probs: List[np.ndarray] = []
    single_test_preds: List[np.ndarray] = []
    single_best: Dict[str, float] = {}

    start_time = time.time()

    ce_loss = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)

    # Train and validate each model
    for model_name in CFG.MODEL_NAMES:
        model, history, metrics, (val_probs, val_preds, val_labels) = train_model(model_name, train_loader, val_loader)
        models.append((model_name, model))
        histories[model_name] = history
        single_val_probs.append(val_probs)
        single_val_preds.append(val_preds)
        single_best[model_name] = metrics["best_val_acc"]

        # Evaluate on test set per model
        if CFG.TTA:
            test_probs, test_preds, test_labels = eval_with_tta(model, test_loader)
            test_loss = 0.0
            test_acc = (test_preds == test_labels).mean()
        else:
            test_loss, test_acc, test_probs, test_preds, test_labels = evaluate(model, test_loader, ce_loss)
        print(f"Test ({model_name}) -> loss={test_loss:.4f} acc={test_acc:.4f}")
        single_test_probs.append(test_probs)
        single_test_preds.append(test_preds)

    print("\nSingle-model best validation accuracies:")
    for k, v in single_best.items():
        print(f"- {k}: {v:.4f}")

    # Ensemble on validation
    val_probs_ens = np.mean(single_val_probs, axis=0)
    val_preds_ens = val_probs_ens.argmax(axis=1)
    val_acc = accuracy_score(val_labels, val_preds_ens)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds_ens, average="macro", zero_division=0)
    print("\nEnsemble Validation Metrics:")
    print(f"Acc={val_acc:.4f} | Prec={val_prec:.4f} | Rec={val_rec:.4f} | F1={val_f1:.4f}")
    prefix_val = f"{CFG.OUTPUT_DIR}/ensemble_val"
    plot_confusion_matrix_and_report(val_labels, val_preds_ens, CLASS_NAMES, prefix_val)
    plot_multiclass_roc(val_labels, val_probs_ens, CLASS_NAMES, f"{prefix_val}_roc.png")

    # Ensemble on test
    test_probs_ens = np.mean(single_test_probs, axis=0)
    test_preds_ens = test_probs_ens.argmax(axis=1)
    test_acc = accuracy_score(test_labels, test_preds_ens)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds_ens, average="macro", zero_division=0)
    print("\nEnsemble Test Metrics:")
    print(f"Acc={test_acc:.4f} | Prec={test_prec:.4f} | Rec={test_rec:.4f} | F1={test_f1:.4f}")
    prefix_test = f"{CFG.OUTPUT_DIR}/ensemble_test"
    plot_confusion_matrix_and_report(test_labels, test_preds_ens, CLASS_NAMES, prefix_test)
    plot_multiclass_roc(test_labels, test_probs_ens, CLASS_NAMES, f"{prefix_test}_roc.png")

    # Save summary JSON
    summary = {
        "class_names": CLASS_NAMES,
        "single_models_best_val_acc": single_best,
        "ensemble_val": {
            "accuracy": float(val_acc),
            "precision_macro": float(val_prec),
            "recall_macro": float(val_rec),
            "f1_macro": float(val_f1),
        },
        "ensemble_test": {
            "accuracy": float(test_acc),
            "precision_macro": float(test_prec),
            "recall_macro": float(test_rec),
            "f1_macro": float(test_f1),
        },
    }
    with open(f"{CFG.OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min. Outputs saved to {CFG.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

