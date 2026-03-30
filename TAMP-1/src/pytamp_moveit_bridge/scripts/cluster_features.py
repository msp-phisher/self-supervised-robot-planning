#!/usr/bin/env python3
"""
Step 4 & 5: Feature Extraction and K-Means Clustering for TAMP Perception.

This script:
1. Loads the trained SimCLR backbone (from Step 3).
2. Passes all images in the dataset through the backbone to extract 512-D features.
3. Performs K-Means clustering (K=3) on the feature vectors.
4. Manually maps cluster IDs to cube identities (red, green, blue).
5. Saves the K-Means model and the identity mapping.

Usage:
    python3 cluster_features.py --model models/simclr_backbone.pt
"""

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.cluster import KMeans
import joblib # type: ignore
import pickle

# ──────────────────────────────────────────────────────────────────────────────
# 1. SETUP
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'images')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
IMAGE_SIZE = 224

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATASET (Simple retrieval, no augmentation)
# ──────────────────────────────────────────────────────────────────────────────

class SimpleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

# ──────────────────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_object_crops(image_path):
    """Detect boxes in full image and return list of cropped images using HSV masking."""
    cv_img = cv2.imread(image_path)
    if cv_img is None: return []
    h, w, _ = cv_img.shape
    
    # Use HSV to find vibrant objects (Red, Green, Blue) and ignore gray/white
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    # Mask for high saturation (vibrant colors only)
    mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crops = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 10000:
            x, y, wb, hb = cv2.boundingRect(cnt)
            # Add dynamic margin
            m = 5
            x1, y1 = max(0, x-m), max(0, y-m)
            x2, y2 = min(w, x+wb+m), min(h, y+hb+m)
            crop = cv_img[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return crops

def extract_features(model_path, device):
    """Load backbone and extract features from object crops in all images."""
    # 1. Load backbone
    backbone = models.resnet18()
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.load_state_dict(torch.load(model_path, map_location=device))
    backbone.to(device).eval()

    # 2. Transform for crops
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_image_paths = sorted([
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(('.png', '.jpg'))
    ])

    features = []
    crop_info = [] # store (original_path, index_in_image)

    print(f"Detecting objects and extracting features from {len(all_image_paths)} images...")
    
    with torch.no_grad():
        for path in all_image_paths:
            crops = extract_object_crops(path)
            if not crops: continue
            
            # 3. Backbone Features
            crop_tensors = torch.stack([transform(c) for c in crops]).to(device)
            feats = backbone(crop_tensors).squeeze(-1).squeeze(-1).cpu().numpy()
            
            # 4. Color Features (Mean RGB)
            color_feats = []
            for c in crops:
                mean_rgb = np.mean(c, axis=(0, 1)) / 255.0
                color_feats.append(mean_rgb * 1000.0) # Massive weight for strict color identity
            color_feats = np.stack(color_feats)
            
            # Hybrid Feature
            hybrid = np.hstack([feats, color_feats])
            features.append(hybrid)
            
            for i in range(len(crops)):
                crop_info.append((path, i))

    return np.vstack(features), crop_info

# ──────────────────────────────────────────────────────────────────────────────
# 4. CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────

def perform_clustering(features, paths, k=3):
    """Run K-Means and save the result."""
    print(f"Running K-Means clustering (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features)

    # Let's inspect a few images from each cluster to help the user map them
    print("\n" + "="*40)
    print(" CLUSTER INSPECTION (Sample Images)")
    print("="*40)
    
    clusters = {i: [] for i in range(k)}
    for cid, (path, idx) in zip(cluster_ids, paths):
        clusters[cid].append(f"{os.path.basename(path)}#{idx}")

    for cid in range(k):
        samples = clusters[cid][:15]
        print(f"Cluster {cid}: {', '.join(samples)} ... (count: {len(clusters[cid])})")
    
    print("="*40 + "\n")
    
    return kmeans

# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=os.path.join(MODEL_DIR, 'simclr_backbone.pt'))
    parser.add_argument('--k', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Extract
    features, paths = extract_features(args.model, device)
    
    # 2. Cluster
    kmeans = perform_clustering(features, paths, k=args.k)
    
    # 3. Save K-Means model
    kmeans_path = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Saved K-Means model to {kmeans_path}")

    # Note: Mapping clusters to colors requires human inspection.
    # In Step 5 of the goal, we'll ask the user to confirm:
    # cluster 0 -> red, cluster 1 -> green, cluster 2 -> blue (or similar)
    print("\nNext step: Map Cluster IDs to symbolic labels (Red, Green, Blue).")
    print("See the sample filenames above to decide.")

if __name__ == '__main__':
    main()
