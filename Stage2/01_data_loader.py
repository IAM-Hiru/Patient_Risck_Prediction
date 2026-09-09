"""
================================================================================
STAGE 02 MULTI-MODAL CANCER PROGRESSION PIPELINE - DATA ENGINEERING MODULE
File: 01_data_loader.py
Role: Data Engineer
================================================================================
PyTorch sequence data loader module for multi-modal cancer progression prediction.
Handles sequence data extraction from CSV, 2D image loading (histology & CT), 
tabular feature scaling, and patient-level train/val/test splitting.
"""

import os
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler


class Stage02MultiModalDataset(Dataset):
    """
    PyTorch Dataset for loading multi-modal longitudinal cancer progression data.
    
    Loads:
      - Histopathology image tile: [3, 224, 224]
      - 2D CT Scan slice:          [1, 224, 224]
      - Tabular biomarkers:        [4] (ctDNA VAF %, Tumor Vol, Biomarker, WSI Atypia)
      - Progression label:         [1] or sequence [seq_len]
    """

    TABULAR_COLS = ["ctDNA_vaf_pct", "ct_tumor_vol_cm3", "biomarker_ng_ml", "wsi_atypia_score"]

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        seq_len: int = 5,
        transform_histo: Optional[transforms.Compose] = None,
        transform_ct: Optional[transforms.Compose] = None,
        scaler: Optional[StandardScaler] = None,
        is_train: bool = False
    ):
        """
        Args:
            df: Filtered DataFrame containing patient sequence records.
            img_dir: Root directory for 2D images (e.g., './2d_images_large').
            seq_len: Number of sequence timepoints per patient (default: 5).
            transform_histo: PyTorch torchvision transform for 3-channel histology images.
            transform_ct: PyTorch torchvision transform for 1-channel CT slice images.
            scaler: Fitted sklearn StandardScaler for tabular normalization.
            is_train: Flag indicating whether this split is for training.
        """
        self.df = df.copy()
        self.img_dir = img_dir
        self.seq_len = seq_len

        # Default transforms for [3, 224, 224] histology and [1, 224, 224] CT slices
        self.transform_histo = transform_histo or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_ct = transform_ct or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Group data by patient_id
        self.patient_ids = self.df["patient_id"].unique()
        self.patient_groups: Dict[str, pd.DataFrame] = {
            pid: group.sort_values("sequence_step") for pid, group in self.df.groupby("patient_id")
        }

        # Tabular scaling
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            if is_train:
                tabular_raw = self.df[self.TABULAR_COLS].values
                self.scaler.fit(tabular_raw)

    def _load_image(self, rel_path: str, mode: str) -> Image.Image:
        """Helper to resolve and load image from filesystem safely."""
        full_path = os.path.join(self.img_dir, rel_path)
        if not os.path.exists(full_path):
            # Fallback search inside subdirectories
            basename = os.path.basename(rel_path)
            alt_path = None
            for root, _, files in os.walk(self.img_dir):
                if basename in files:
                    alt_path = os.path.join(root, basename)
                    break
            if alt_path and os.path.exists(alt_path):
                full_path = alt_path
            else:
                # Return synthetic placeholder Image if file missing
                if mode == "RGB":
                    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                else:
                    return Image.fromarray(np.zeros((224, 224), dtype=np.uint8))

        return Image.open(full_path).convert(mode)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        patient_df = self.patient_groups[pid]

        histo_tensors = []
        ct_tensors = []
        tabular_list = []
        label_list = []

        cancer_type = patient_df["cancer_type"].iloc[0]

        for _, row in patient_df.iterrows():
            # 1. Load Histopathology tile -> [3, 224, 224]
            histo_img = self._load_image(row["pathology_img_file"], mode="RGB")
            histo_tensor = self.transform_histo(histo_img) # shape: [3, 224, 224]
            histo_tensors.append(histo_tensor)

            # 2. Load CT Slice -> [1, 224, 224]
            ct_img = self._load_image(row["ct_img_file"], mode="L")
            ct_tensor = self.transform_ct(ct_img)          # shape: [1, 224, 224]
            ct_tensors.append(ct_tensor)

            # 3. Tabular Biomarkers -> [4]
            raw_tab = row[self.TABULAR_COLS].values.astype(np.float32).reshape(1, -1)
            norm_tab = self.scaler.transform(raw_tab).squeeze(0) # shape: [4]
            tabular_list.append(torch.tensor(norm_tab, dtype=torch.float32))

            # 4. Progression Label -> scalar float
            label_list.append(float(row["label_progression"]))

        # Stack sequence dimension -> [seq_len, ...]
        pathology_seq = torch.stack(histo_tensors, dim=0) # [5, 3, 224, 224]
        ct_seq = torch.stack(ct_tensors, dim=0)          # [5, 1, 224, 224]
        tabular_seq = torch.stack(tabular_list, dim=0)    # [5, 4]
        labels_seq = torch.tensor(label_list, dtype=torch.float32) # [5]

        return {
            "patient_id": pid,
            "cancer_type": cancer_type,
            "pathology_seq": pathology_seq,  # Shape: [T, 3, 224, 224]
            "ct_seq": ct_seq,                # Shape: [T, 1, 224, 224]
            "tabular_seq": tabular_seq,      # Shape: [T, 4]
            "labels_seq": labels_seq         # Shape: [T]
        }


def get_dataloaders(
    csv_path: str,
    img_dir: str,
    batch_size: int = 16,
    seq_len: int = 5,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Splits data at the PATIENT level (70% train, 15% val, 15% test) to prevent sequence leakage,
    and returns PyTorch DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, fitted_scaler)
    """
    df = pd.read_csv(csv_path)

    # Patient-level split
    unique_patients = np.array(df["patient_id"].unique())
    np.random.seed(random_seed)
    np.random.shuffle(unique_patients)

    n_total = len(unique_patients)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    train_pids = set(unique_patients[:n_train])
    val_pids = set(unique_patients[n_train:n_train + n_val])
    test_pids = set(unique_patients[n_train + n_val:])

    train_df = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_pids)].reset_index(drop=True)

    # Fit scaler on training set
    scaler = StandardScaler()
    scaler.fit(train_df[Stage02MultiModalDataset.TABULAR_COLS].values)

    train_ds = Stage02MultiModalDataset(train_df, img_dir, seq_len=seq_len, scaler=scaler, is_train=True)
    val_ds = Stage02MultiModalDataset(val_df, img_dir, seq_len=seq_len, scaler=scaler, is_train=False)
    test_ds = Stage02MultiModalDataset(test_df, img_dir, seq_len=seq_len, scaler=scaler, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "data", "stage02_lstm_8types_3000samples.csv")
    images_dir = os.path.join(script_dir, "2d_images_large")

    print(f"--- Testing Data Loader Module ---")
    print(f"CSV Path: {csv_file}")
    print(f"Images Dir: {images_dir}")

    if os.path.exists(csv_file):
        train_ldr, val_ldr, test_ldr, scaler = get_dataloaders(csv_file, images_dir, batch_size=4)
        print(f"Split sizes (patients): Train={len(train_ldr.dataset)}, Val={len(val_ldr.dataset)}, Test={len(test_ldr.dataset)}")
        
        batch = next(iter(train_ldr))
        print("\nSequence Batch Shapes:")
        print(f"  Pathology Sequence Tensor : {batch['pathology_seq'].shape}  # [B, T, C, H, W]")
        print(f"  CT Sequence Tensor        : {batch['ct_seq'].shape}         # [B, T, C, H, W]")
        print(f"  Tabular Sequence Tensor   : {batch['tabular_seq'].shape}      # [B, T, Num_Features]")
        print(f"  Labels Sequence Tensor    : {batch['labels_seq'].shape}       # [B, T]")
    else:
        print(f"CSV file not found at {csv_file}. Please check path.")
