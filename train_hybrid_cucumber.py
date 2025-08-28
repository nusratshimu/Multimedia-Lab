import os
import math
import time
import random
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except Exception:
    HAS_ALB = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class CFG:
    data_dir: str = '/kaggle/input/cucumber-disease-dataset'
    csv_path: Optional[str] = None
    output_dir: str = './outputs'
    img_size: int = 224
    num_workers: int = 2
    valid_size: float = 0.2
    seed: int = 42

    # Train
    epochs: int = 25
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    use_pretrained: bool = True
    freeze_backbone_epochs: int = 0
    dropout: float = 0.3
    grad_clip_norm: float = 1.0
    accumulate_steps: int = 1

    # Mixup/CutMix
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.8
    mixup_prob: float = 0.5
    cutmix_prob: float = 0.3

    # Scheduler
    use_one_cycle: bool = True
    max_lr: float = 2e-4
    pct_start: float = 0.3

    # Early stopping
    es_patience: int = 6

    # Model
    backbone_a: str = 'resnet50'
    backbone_b: str = 'efficientnet_b0'


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])

        if HAS_ALB and isinstance(self.transform, A.BasicTransform):
            image_np = np.array(image)
            out = self.transform(image=image_np)
            image_t = out['image']
        else:
            if self.transform is not None:
                image_t = self.transform(image)
            else:
                image_t = transforms.ToTensor()(image)
        return image_t, label


def get_transforms(img_size: int):
    if HAS_ALB:
        train_tfms = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
            A.ColorJitter(0.2,0.2,0.2,0.1, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=int(img_size*0.1), max_width=int(img_size*0.1), p=0.5),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
        valid_tfms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
        valid_tfms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
    return train_tfms, valid_tfms


def discover_from_folder(root_dir: str) -> pd.DataFrame:
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    records = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):
                records.append({'image_path': os.path.join(class_name, fname), 'label': class_idx})
    df = pd.DataFrame(records)
    return df, class_names


def stratified_split(df: pd.DataFrame, valid_size: float, seed: int = 42):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    y = df['label'].values
    for train_idx, valid_idx in splitter.split(df, y):
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)


def mixup_cutmix(images, targets, mixup_alpha: float, cutmix_alpha: float, mixup_prob: float, cutmix_prob: float):
    batch_size = images.size(0)
    device = images.device
    lam = 1.0
    shuffled_indices = torch.randperm(batch_size).to(device)
    images2 = images[shuffled_indices]
    targets2 = targets[shuffled_indices]

    r = random.random()
    if r < cutmix_prob and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        W = images.size(3)
        H = images.size(2)
        cut_w = int(W * math.sqrt(1 - lam))
        cut_h = int(H * math.sqrt(1 - lam))
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        images[:, :, y1:y2, x1:x2] = images2[:, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        return images, targets, targets2, lam
    elif r < mixup_prob + cutmix_prob and mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = images * lam + images2 * (1 - lam)
        return images, targets, targets2, lam
    else:
        return images, targets, targets, 1.0


class HybridBackbone(nn.Module):
    def __init__(self, backbone_a: str, backbone_b: str, pretrained: bool = True, dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        # A: ResNet family
        if backbone_a.startswith('resnet'):
            model_a = getattr(torchvision.models, backbone_a)(weights=(torchvision.models.ResNet50_Weights.DEFAULT if pretrained and backbone_a=='resnet50' else None))
            in_features_a = model_a.fc.in_features
            model_a.fc = nn.Identity()
            self.backbone_a = model_a
        else:
            raise ValueError('Unsupported backbone_a')

        # B: EfficientNet family
        if backbone_b.startswith('efficientnet'):
            weights_b = None
            if pretrained:
                if backbone_b == 'efficientnet_b0':
                    weights_b = torchvision.models.EfficientNet_B0_Weights.DEFAULT
                elif backbone_b == 'efficientnet_b1':
                    weights_b = torchvision.models.EfficientNet_B1_Weights.DEFAULT
            model_b = getattr(torchvision.models, backbone_b)(weights=weights_b)
            in_features_b = model_b.classifier[-1].in_features
            model_b.classifier[-1] = nn.Identity()
            self.backbone_b = model_b
        else:
            raise ValueError('Unsupported backbone_b')

        concat_dim = in_features_a + in_features_b
        self.head = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Dropout(dropout),
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(concat_dim // 2, num_classes)
        )

    def forward(self, x):
        xa = self.backbone_a(x)
        xb = self.backbone_b(x)
        xcat = torch.cat([xa, xb], dim=1)
        return self.head(xcat)


def label_smoothing_cross_entropy(pred: torch.Tensor, target: torch.Tensor, smoothing: float, num_classes: int):
    if smoothing <= 0:
        return F.cross_entropy(pred, target)
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1 - smoothing)
    return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, cfg: CFG, num_classes: int, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        images_aug, t1, t2, lam = mixup_cutmix(images, targets, cfg.mixup_alpha, cfg.cutmix_alpha, cfg.mixup_prob, cfg.cutmix_prob)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images_aug)
            if lam != 1.0:
                loss = lam * label_smoothing_cross_entropy(outputs, t1, cfg.label_smoothing, num_classes) + \
                       (1 - lam) * label_smoothing_cross_entropy(outputs, t2, cfg.label_smoothing, num_classes)
            else:
                loss = label_smoothing_cross_entropy(outputs, t1, cfg.label_smoothing, num_classes)

        scaler.scale(loss).backward()
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        with torch.no_grad():
            preds = outputs.argmax(1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)
        running_loss += loss.item() * images.size(0)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    return val_loss, val_acc, y_true, y_prob


def plot_curves(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curves')
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curves')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=200)
    plt.close()


def evaluate_and_plot(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()

    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        if y_true_bin.shape[1] == 1:
            auc = roc_auc_score(y_true, y_prob[:,1])
            fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr, label=f'ROC AUC={auc:.4f}')
            plt.plot([0,1],[0,1],'--', color='gray')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('ROC Curve')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=200)
            plt.close()
        else:
            auc = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
            plt.figure(figsize=(6,5))
            for i in range(y_true_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_prob[:,i])
                plt.plot(fpr, tpr, label=f'{class_names[i]}')
            plt.plot([0,1],[0,1],'--', color='gray')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Curves (AUC_w={auc:.4f})'); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'roc_curves.png'), dpi=200)
            plt.close()
    except Exception:
        auc = float('nan')

    metrics = {
        'accuracy': float(acc),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'roc_auc_weighted': float(auc) if not math.isnan(auc) else None
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def build_model(num_classes: int, cfg: CFG, device):
    model = HybridBackbone(cfg.backbone_a, cfg.backbone_b, pretrained=cfg.use_pretrained, dropout=cfg.dropout, num_classes=num_classes)
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=CFG.data_dir)
    parser.add_argument('--csv_path', type=str, default=CFG.csv_path)
    parser.add_argument('--output_dir', type=str, default=CFG.output_dir)
    parser.add_argument('--img_size', type=int, default=CFG.img_size)
    parser.add_argument('--epochs', type=int, default=CFG.epochs)
    parser.add_argument('--batch_size', type=int, default=CFG.batch_size)
    parser.add_argument('--lr', type=float, default=CFG.lr)
    parser.add_argument('--weight_decay', type=float, default=CFG.weight_decay)
    parser.add_argument('--label_smoothing', type=float, default=CFG.label_smoothing)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--no_pretrained', dest='use_pretrained', action='store_false')
    parser.set_defaults(use_pretrained=CFG.use_pretrained)
    parser.add_argument('--dropout', type=float, default=CFG.dropout)
    parser.add_argument('--mixup_alpha', type=float, default=CFG.mixup_alpha)
    parser.add_argument('--cutmix_alpha', type=float, default=CFG.cutmix_alpha)
    parser.add_argument('--mixup_prob', type=float, default=CFG.mixup_prob)
    parser.add_argument('--cutmix_prob', type=float, default=CFG.cutmix_prob)
    parser.add_argument('--use_one_cycle', action='store_true')
    parser.add_argument('--no_one_cycle', dest='use_one_cycle', action='store_false')
    parser.set_defaults(use_one_cycle=CFG.use_one_cycle)
    parser.add_argument('--max_lr', type=float, default=CFG.max_lr)
    parser.add_argument('--pct_start', type=float, default=CFG.pct_start)
    parser.add_argument('--es_patience', type=int, default=CFG.es_patience)
    parser.add_argument('--backbone_a', type=str, default=CFG.backbone_a)
    parser.add_argument('--backbone_b', type=str, default=CFG.backbone_b)
    args = parser.parse_args()

    set_seed(CFG.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    if args.csv_path and os.path.isfile(args.csv_path):
        df = pd.read_csv(args.csv_path)
        assert {'image_path','label'}.issubset(df.columns)
        class_names = sorted(df['label'].unique().tolist())
        class_map = {label: idx for idx, label in enumerate(class_names)}
        if not np.issubdtype(df['label'].dtype, np.integer):
            df['label'] = df['label'].map(class_map)
    else:
        df, class_names = discover_from_folder(args.data_dir)

    num_classes = len(class_names)
    train_df, valid_df = stratified_split(df, valid_size=CFG.valid_size, seed=CFG.seed)

    train_tfms, valid_tfms = get_transforms(args.img_size)
    train_ds = ImageDataset(train_df, image_root=args.data_dir, transform=train_tfms)
    valid_ds = ImageDataset(valid_df, image_root=args.data_dir, transform=valid_tfms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

    # Model
    model = HybridBackbone(args.backbone_a, args.backbone_b, pretrained=args.use_pretrained, dropout=args.dropout, num_classes=num_classes)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    if args.use_one_cycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps, pct_start=args.pct_start)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    best_path = os.path.join(args.output_dir, 'best_model.pt')
    es_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, CFG, num_classes, scheduler)
        val_loss, val_acc, y_true, y_prob = validate(model, valid_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={(time.time()-t0):.1f}s')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'class_names': class_names}, best_path)
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                print('Early stopping triggered')
                break

    plot_curves(history, args.output_dir)

    # Load best model and final eval
    ckpt = torch.load(best_path, map_location='cpu') if os.path.isfile(best_path) else None
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
    val_loss, val_acc, y_true, y_prob = validate(model, valid_loader, device)
    metrics = evaluate_and_plot(y_true, y_prob, class_names, args.output_dir)
    print('Final metrics:', json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

