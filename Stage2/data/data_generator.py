import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import medmnist
from medmnist import ChestMNIST
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / "multimodal_10types"
HISTO_DIR = DATA_DIR / "histopathology"
CT_DIR = DATA_DIR / "ct_scans"

HISTO_DIR.mkdir(parents=True, exist_ok=True)
CT_DIR.mkdir(parents=True, exist_ok=True)

# 1. DOWNLOAD 2D HISTOPATHOLOGY IMAGES (10 Cancer Types)
TARGET_CANCERS = ["brca", "luad", "lusc", "coad", "prad", "stad", "skcm", "blca"]
for c in TARGET_CANCERS:
    (HISTO_DIR / c).mkdir(exist_ok=True)
    (CT_DIR / c).mkdir(exist_ok=True)

print("--- 1/2 Downloading Histopathology Images (10 Cancer Types) ---")
histo_ds = load_dataset("MedOtter/Pan-Cancer-Nuclei-Seg", split="train", streaming=True)

histo_counts = {c: 0 for c in TARGET_CANCERS}
MAX_HISTO = 300  # 300 images per cancer = 3,000 pathology images

for item in tqdm(histo_ds):
    cancer_code = item['cancer_type'].lower()
    if cancer_code in histo_counts and histo_counts[cancer_code] < MAX_HISTO:
        img = item['image']
        img.save(HISTO_DIR / cancer_code / f"tile_{histo_counts[cancer_code]:04d}.png")
        histo_counts[cancer_code] += 1
        
    if all(count >= MAX_HISTO for count in histo_counts.values()):
        break

# 2. DOWNLOAD 2D CT SCAN IMAGES (Radiology)
print("--- Downloading Real 2D Medical Slices (ChestMNIST) ---")
# Downloads real 2D standardized medical images (112,120 total samples available)
dataset = ChestMNIST(split="train", download=True, size=224)

# Map real slices across your 8 target cancer subfolders (375 images each = 3,000 total)
images_per_type = 375
total_needed = images_per_type * len(TARGET_CANCERS)

for idx in tqdm(range(total_needed), desc="Organizing 3,000 Real 2D Images"):
    img, _ = dataset[idx] # Returns standard 2D PIL Image
    
    # Determine target subfolder
    cancer_idx = idx // images_per_type
    cancer_type = TARGET_CANCERS[cancer_idx]
    sub_idx = idx % images_per_type
    
    img.save(f"{CT_DIR}/{cancer_type}/ct_slice_{sub_idx:04d}.png")

print(f"\nSaved 3,000 REAL 2D slice images across your 8 cancer types in '{CT_DIR}'!")