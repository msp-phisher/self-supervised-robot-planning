#!/usr/bin/env python3
import os
import cv2
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'images')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
IMAGE_SIZE = 224

def extract_object_crops(image_path):
    cv_img = cv2.imread(image_path)
    if cv_img is None: return []
    h, w, _ = cv_img.shape
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    # Mask for high saturation (vibrant colors only)
    mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 10000:
            x, y, wb, hb = cv2.boundingRect(cnt)
            m = 5
            x1, y1 = max(0, x-m), max(0, y-m)
            x2, y2 = min(w, x+wb+m), min(h, y+hb+m)
            crop = cv_img[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return crops

def main():
    kmeans_path = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
    backbone_path = os.path.join(MODEL_DIR, 'simclr_backbone.pt')
    
    with open(kmeans_path, 'rb') as f:
        kmeans = pickle.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = models.resnet18()
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    backbone.to(device).eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_image_paths = sorted([os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg'))])
    
    # Store a few crops for each predicted cluster
    k = kmeans.n_clusters
    cluster_samples = {i: [] for i in range(k)}
    
    print("Collecting samples for visualization...")
    for path in all_image_paths[:200]: # check first 200 images
        crops = extract_object_crops(path)
        if not crops: continue
        
        with torch.no_grad():
            crop_tensors = torch.stack([transform(c) for c in crops]).to(device)
            feats = backbone(crop_tensors).squeeze(-1).squeeze(-1).cpu().numpy()
            
            # Combine with Color Features (Mean RGB)
            color_feats = []
            for c in crops:
                mean_rgb = np.mean(c, axis=(0, 1)) / 255.0
                color_feats.append(mean_rgb * 1000.0)
            color_feats = np.stack(color_feats)
            
            hybrid = np.hstack([feats, color_feats])
            
        preds = kmeans.predict(hybrid)
        for crop, cid in zip(crops, preds):
            if len(cluster_samples[cid]) < 25:
                cluster_samples[cid].append(crop)

    # Save montages
    for cid in range(k):
        samples = cluster_samples[cid]
        if not samples: continue
        
        # Create a 5x5 grid
        grid_size = 5
        cell_size = 100
        montage = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)
        
        for idx, s in enumerate(samples[:25]):
            r, c = idx // grid_size, idx % grid_size
            s_resized = cv2.resize(cv2.cvtColor(s, cv2.COLOR_RGB2BGR), (cell_size, cell_size))
            montage[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = s_resized
            
        out_path = os.path.join(PROJECT_ROOT, f'cluster_{cid}_samples.png')
        cv2.imwrite(out_path, montage)
        print(f"Saved cluster {cid} samples to {out_path}")

if __name__ == '__main__':
    main()
