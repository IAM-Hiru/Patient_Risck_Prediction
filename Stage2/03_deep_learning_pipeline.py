"""
================================================================================
STAGE 02 MULTI-MODAL CANCER PROGRESSION PIPELINE - DEEP LEARNING MODULE
File: 03_deep_learning_pipeline.py
Role: Deep Learning Engineer
================================================================================
PyTorch Multi-Modal Architecture & Longitudinal Training Pipeline.
Combines:
  1. PathologyEncoder (3-channel ResNet-18/CNN -> 128-dim embedding)
  2. CTEncoder (1-channel CNN -> 128-dim embedding)
  3. MultimodalLSTM (2-layer Bidirectional LSTM accepting 260-dim input -> 256-dim output -> Classification Head)

Trains using BCEWithLogitsLoss, AdamW optimizer, and ReduceLROnPlateau scheduler.
Saves best model weights to 'best_stage02_multimodal_lstm.pth'.
"""

import os
import time
from typing import Dict, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet18_Weights

import importlib.util
script_dir = os.path.dirname(os.path.abspath(__file__))
data_loader_path = os.path.join(script_dir, "01_data_loader.py")
spec_dl = importlib.util.spec_from_file_location("loader_mod", data_loader_path)
dl_mod = importlib.util.module_from_spec(spec_dl)
spec_dl.loader.exec_module(dl_mod)
get_dataloaders = dl_mod.get_dataloaders


# ============================================================
# 1. VISION ENCODERS
# ============================================================

class PathologyEncoder(nn.Module):
    """
    Encoder for 3-channel Histopathology tile images [B, 3, 224, 224].
    Uses ResNet-18 feature extractor mapping to a 128-dimensional embedding.
    """
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super(PathologyEncoder, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # Remove classifier head
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B*T, 3, 224, 224]
        feats = self.backbone(x)          # [B*T, 512]
        embeds = self.fc(feats)           # [B*T, 128]
        return embeds


class CTEncoder(nn.Module):
    """
    Custom 2D Convolutional Encoder for 1-channel CT slice images [B, 1, 224, 224].
    Maps input to a 128-dimensional embedding.
    """
    def __init__(self, embedding_dim: int = 128):
        super(CTEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [32, 112, 112]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [64, 56, 56]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# [128, 28, 28]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))                          # [128, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B*T, 1, 224, 224]
        feats = self.conv(x)              # [B*T, 128, 4, 4]
        feats = feats.view(feats.size(0), -1) # [B*T, 2048]
        embeds = self.fc(feats)           # [B*T, 128]
        return embeds


# ============================================================
# 2. MULTI-MODAL SEQUENCE LSTM ARCHITECTURE
# ============================================================

class MultimodalLSTM(nn.Module):
    """
    End-to-End Multi-Modal Longitudinal Sequence Model.
    
    Inputs:
      - pathology_seq : [B, T, 3, 224, 224]
      - ct_seq        : [B, T, 1, 224, 224]
      - tabular_seq   : [B, T, 4]
      
    Concatenation per timepoint:
      128 (Pathology) + 128 (CT) + 4 (Tabular) = 260 dim.
      
    Bidirectional LSTM:
      hidden_dim = 128, num_layers = 2, dropout = 0.3 -> output dim 256.
      
    Output:
      Logits per timepoint [B, T] for binary progression prediction.
    """
    def __init__(
        self,
        histo_embed_dim: int = 128,
        ct_embed_dim: int = 128,
        tabular_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(MultimodalLSTM, self).__init__()

        self.histo_encoder = PathologyEncoder(embedding_dim=histo_embed_dim)
        self.ct_encoder = CTEncoder(embedding_dim=ct_embed_dim)

        input_dim = histo_embed_dim + ct_embed_dim + tabular_dim  # 128 + 128 + 4 = 260

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_out_dim = hidden_dim * 2  # Bidirectional -> 256

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        pathology_seq: torch.Tensor,
        ct_seq: torch.Tensor,
        tabular_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass handling 5-step patient sequences.
        
        Shapes:
          pathology_seq : [B, T, 3, H, W]
          ct_seq        : [B, T, 1, H, W]
          tabular_seq   : [B, T, 4]
        Returns:
          Logits        : [B, T]
        """
        B, T, C_h, H_h, W_h = pathology_seq.shape
        _, _, C_c, H_c, W_c = ct_seq.shape

        # 1. Flatten Batch & Time dimensions for vision feature extraction
        h_flat = pathology_seq.view(B * T, C_h, H_h, W_h)  # [B*T, 3, 224, 224]
        c_flat = ct_seq.view(B * T, C_c, H_c, W_c)         # [B*T, 1, 224, 224]

        h_embeds = self.histo_encoder(h_flat)  # [B*T, 128]
        c_embeds = self.ct_encoder(c_flat)     # [B*T, 128]

        # 2. Reshape embeddings back to sequence format [B, T, dim]
        h_seq = h_embeds.view(B, T, -1)        # [B, T, 128]
        c_seq = c_embeds.view(B, T, -1)        # [B, T, 128]

        # 3. Concatenate vision embeddings + tabular features
        multimodal_seq = torch.cat([h_seq, c_seq, tabular_seq], dim=-1)  # [B, T, 260]

        # 4. Pass through Bidirectional LSTM
        lstm_out, _ = self.lstm(multimodal_seq)  # [B, T, 256]

        # 5. Compute progression logits per timepoint
        logits = self.classifier(lstm_out).squeeze(-1)  # [B, T]

        return logits


# ============================================================
# 3. TRAINING & VALIDATION PIPELINE
# ============================================================

def train_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-4,
    device: Optional[str] = None,
    save_model_path: str = "best_stage02_multimodal_lstm.pth"
) -> Dict[str, list]:
    """
    Executes training loop over specified epochs, logging train loss, val loss, 
    and val accuracy per epoch. Saves best model weights based on val loss.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n============================================================")
    print(f"STARTING DEEP LEARNING MODEL TRAINING PIPELINE")
    print(f"Device: {device} | Epochs: {epochs} | Initial LR: {lr}")
    print(f"============================================================\n")

    model = MultimodalLSTM().to(device)
    # Class imbalance fix: ~73% negative / ~27% positive → pos_weight ≈ 2.7
    # This penalises the model more heavily for missing true progressors (improves Recall & F1)
    pos_weight = torch.tensor([2.7]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            h_seq = batch["pathology_seq"].to(device)
            c_seq = batch["ct_seq"].to(device)
            tab_seq = batch["tabular_seq"].to(device)
            labels = batch["labels_seq"].to(device) # [B, T]

            optimizer.zero_grad()
            logits = model(h_seq, c_seq, tab_seq) # [B, T]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * h_seq.size(0)
            total_samples += h_seq.size(0)

        epoch_train_loss = running_loss / total_samples

        # --- VALIDATION PHASE ---
        model.eval()
        val_running_loss = 0.0
        val_total_samples = 0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for batch in val_loader:
                h_seq = batch["pathology_seq"].to(device)
                c_seq = batch["ct_seq"].to(device)
                tab_seq = batch["tabular_seq"].to(device)
                labels = batch["labels_seq"].to(device)

                logits = model(h_seq, c_seq, tab_seq)
                loss = criterion(logits, labels)

                val_running_loss += loss.item() * h_seq.size(0)
                val_total_samples += h_seq.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.numel()

        epoch_val_loss = val_running_loss / val_total_samples
        epoch_val_acc = correct_preds / total_preds
        elapsed = time.time() - t0

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f} ({epoch_val_acc * 100:.1f}%) | "
              f"LR: {current_lr:.2e}")

        # Checkpoint saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            dir_path = os.path.dirname(save_model_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": epoch_val_acc
            }, save_model_path)
            print(f"  --> Saved new best checkpoint: '{save_model_path}'")

    print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.4f}\n")
    return history


if __name__ == "__main__":
    import importlib.util
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_loader_path = os.path.join(script_dir, "01_data_loader.py")
    spec = importlib.util.spec_from_file_location("data_loader_mod", data_loader_path)
    dl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl_mod)

    csv_file = os.path.join(script_dir, "data", "stage02_lstm_8types_3000samples.csv")
    if not os.path.exists(csv_file):
        csv_file = os.path.join(script_dir, "stage02_lstm_8types_3000samples.csv")
    images_dir = os.path.join(script_dir, "2d_images_large")

    print("Loading data for training...")
    train_ldr, val_ldr, test_ldr, scaler = dl_mod.get_dataloaders(
        csv_file, images_dir, batch_size=16, seq_len=5
    )

    # Train model for epochs
    model_save = os.path.join(script_dir, "best_stage02_multimodal_lstm.pth")
    history = train_pipeline(train_ldr, val_ldr, epochs=3, lr=1e-4, save_model_path=model_save)
