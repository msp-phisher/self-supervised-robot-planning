#!/usr/bin/env python3
"""
Step 3: Self-Supervised SimCLR Training for TAMP Visual Perception.

This script trains a SimCLR contrastive learning model on unlabeled images
collected from the Gazebo camera. No labels are used — the model learns
feature representations purely from augmented image pairs.

Usage:
    python3 train_simclr.py                          # default: 100 epochs
    python3 train_simclr.py --epochs 200 --batch_size 64

Outputs:
    models/simclr_backbone.pt   — trained ResNet-18 encoder (frozen for inference)
    models/simclr_full.pt       — full model with projection head (for resuming)
    models/training_log.csv     — epoch, train_loss, val_loss log
"""

import os
import sys
import csv
import argparse
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'images')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Default hyperparams
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32  # Reduced to prevent OOM
DEFAULT_LR = 3e-4
DEFAULT_TEMPERATURE = 0.1
DEFAULT_PROJ_DIM = 128
IMAGE_SIZE = 224   # ResNet expected input


# ──────────────────────────────────────────────────────────────────────────────
# 2. AUGMENTATION PIPELINE (SimCLR-style)
# ──────────────────────────────────────────────────────────────────────────────

class SimCLRAugmentation:
    """
    Applies two different random augmentations to the same image,
    producing a positive pair (view_i, view_j) for contrastive learning.

    Augmentations:
        - Random resized crop (scale 0.2 – 1.0)
        - Random horizontal flip
        - Color jitter (brightness, contrast, saturation, hue)
        - Random grayscale (p=0.2)
        - Gaussian blur (p=0.5)
        - Normalize to ImageNet stats
    """

    def __init__(self, size=IMAGE_SIZE):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, x):
        """Return two differently augmented views of the same image."""
        return self.transform(x), self.transform(x)


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ──────────────────────────────────────────────────────────────────────────────

class UnlabeledImageDataset(Dataset):
    """
    Loads all PNG/JPG images from a directory — NO LABELS.
    Each __getitem__ returns a pair of augmented views.
    """

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            view_i, view_j = self.transform(img)
        else:
            # Fallback: just resize and tensor
            t = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ])
            view_i = t(img)
            view_j = t(img)
        return view_i, view_j


# ──────────────────────────────────────────────────────────────────────────────
# 4. SIMCLR MODEL
# ──────────────────────────────────────────────────────────────────────────────

class SimCLREncoder(nn.Module):
    """
    SimCLR model = ResNet-18 backbone + MLP projection head.

    The backbone produces a 512-D feature vector (after global avg pool).
    The projection head maps it to a 128-D unit sphere for contrastive loss.

    At inference time, we discard the projection head and use only the
    512-D backbone features for clustering.
    """

    def __init__(self, projection_dim=DEFAULT_PROJ_DIM):
        super().__init__()

        # Backbone: ResNet-18 (lighter than ResNet-50, good for small datasets)
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone_dim = backbone.fc.in_features  # 512

        # Remove the final FC layer — we'll use our own projection head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Projection head: 512 → 256 → 128
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projection_dim),
        )

    def forward(self, x):
        """Returns (features_512d, projections_128d)."""
        h = self.backbone(x)          # [B, 512, 1, 1]
        h = h.squeeze(-1).squeeze(-1) # [B, 512]
        z = self.projection_head(h)   # [B, 128]
        z = F.normalize(z, dim=1)     # L2 normalize onto unit sphere
        return h, z

    def extract_features(self, x):
        """Extract only the 512-D backbone features (for clustering)."""
        h = self.backbone(x)
        h = h.squeeze(-1).squeeze(-1)
        return h


# ──────────────────────────────────────────────────────────────────────────────
# 5. NT-XENT CONTRASTIVE LOSS
# ──────────────────────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    For a batch of N images, SimCLR generates 2N augmented views.
    For each positive pair (i, j), the loss pushes their projections
    together while pushing apart from all 2(N-1) negative pairs.
    """

    def __init__(self, temperature=DEFAULT_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: [B, proj_dim] — L2-normalized projections of two views.
        """
        batch_size = z_i.shape[0]

        # Concatenate both views: [2B, proj_dim]
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix: [2B, 2B]
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=sim.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_i = torch.arange(batch_size, device=sim.device)
        pos_j = pos_i + batch_size

        # Labels: for row i, positive is at column i+B; for row i+B, positive is at column i
        labels = torch.cat([pos_j, pos_i], dim=0)

        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)
        return loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# ──────────────────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train_simclr(args):
    """Main training function."""

    # ── Setup ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  SimCLR Self-Supervised Training")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"{'='*60}\n")

    # Reproducibility
    torch.manual_seed(42)

    # ── Dataset ──
    augmentation = SimCLRAugmentation(size=IMAGE_SIZE)
    full_dataset = UnlabeledImageDataset(DATASET_DIR, transform=augmentation)
    total = len(full_dataset)
    
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True
    )

    # ── Model ──
    model = SimCLREncoder(projection_dim=args.proj_dim).to(device)
    criterion = NTXentLoss(temperature=args.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training log ──
    log_path = os.path.join(MODEL_DIR, 'training_log.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc_top1', 'val_acc_top1', 'lr'])

    best_val_loss = float('inf')

    # ── Epoch loop ──
    for epoch in range(1, args.epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        for view_i, view_j in train_loader:
            view_i, view_j = view_i.to(device), view_j.to(device)
            _, z_i = model(view_i)
            _, z_j = model(view_j)

            # --- LOSS CALCULATION ---
            # (Note: NTXentLoss internal sim matrix calculation)
            batch_size = z_i.shape[0]
            z = torch.cat([z_i, z_j], dim=0)
            sim = torch.mm(z, z.t()) / args.temperature
            mask = torch.eye(2 * batch_size, device=sim.device).bool()
            sim.masked_fill_(mask, -1e9)
            pos_i = torch.arange(batch_size, device=sim.device)
            pos_j = pos_i + batch_size
            labels = torch.cat([pos_j, pos_i], dim=0)
            
            loss = F.cross_entropy(sim, labels)
            acc1, _ = accuracy(sim, labels, topk=(1, 5))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += acc1.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches
        train_acc = train_acc_sum / train_batches

        # --- VALIDATE ---
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for view_i, view_j in val_loader:
                view_i, view_j = view_i.to(device), view_j.to(device)
                _, z_i = model(view_i)
                _, z_j = model(view_j)

                batch_size = z_i.shape[0]
                z = torch.cat([z_i, z_j], dim=0)
                sim = torch.mm(z, z.t()) / args.temperature
                mask = torch.eye(2 * batch_size, device=sim.device).bool()
                sim.masked_fill_(mask, -1e9)
                labels = torch.cat([torch.arange(batch_size, device=sim.device) + batch_size, 
                                   torch.arange(batch_size, device=sim.device)], dim=0)

                loss = F.cross_entropy(sim, labels)
                acc1, _ = accuracy(sim, labels, topk=(1, 5))
                
                val_loss_sum += loss.item()
                val_acc_sum += acc1.item()
                val_batches += 1

        val_loss = val_loss_sum / val_batches
        val_acc = val_acc_sum / val_batches
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        log_writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}', f'{train_acc:.2f}', f'{val_acc:.2f}', f'{current_lr:.6f}'])
        log_file.flush()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | Acc@1: {train_acc:.2f}% | Val Acc@1: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.backbone.state_dict(), os.path.join(MODEL_DIR, 'simclr_backbone.pt'))

    log_file.close()
    print(f"\n  Training complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model saved to {MODEL_DIR}/simclr_backbone.pt")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Models saved:   {MODEL_DIR}/")
    print(f"    simclr_backbone.pt      — best backbone (for clustering)")
    print(f"    simclr_full.pt          — best full model (for resume)")
    print(f"    simclr_backbone_final.pt — final backbone")
    print(f"    simclr_full_final.pt    — final full model")
    print(f"  Training log:   {log_path}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SimCLR Self-Supervised Training for TAMP Perception')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help=f'Learning rate (default: {DEFAULT_LR})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'NT-Xent temperature (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--proj_dim', type=int, default=DEFAULT_PROJ_DIM,
                        help=f'Projection head output dim (default: {DEFAULT_PROJ_DIM})')
    args = parser.parse_args()

    train_simclr(args)


if __name__ == '__main__':
    main()
