"""
================================================================================
STAGE 02 MULTI-MODAL CANCER PROGRESSION PIPELINE - INTEGRATION MODULE
File: 05_pipeline_integration.py
Role: Integration Engineer
================================================================================
Single-patient end-to-end inference and deployment wrapper.
Processes a patient's multi-modal 5-timepoint longitudinal trajectory:
  - Extracts 2D pathology & CT image embeddings
  - Normalizes biomarkers using fitted StandardScaler
  - Passes sequence through trained MultimodalLSTM model
  - Outputs per-step and trajectory progression probabilities
  - Assigns Clinical Risk Tiers:
      * "Low Progression Risk"    (< 0.35)
      * "Moderate Risk"           (0.35 - 0.70)
      * "High Progression Risk"   (> 0.70)

Includes execution block demonstrating inference on patient 'PAT-0001'.
"""

import os
import importlib.util
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler


def load_model_class(script_dir: str):
    """Dynamically imports MultimodalLSTM from deep_learning_pipeline module."""
    model_path = os.path.join(script_dir, "03_deep_learning_pipeline.py")
    spec_model = importlib.util.spec_from_file_location("model_mod", model_path)
    model_mod = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(model_mod)
    return model_mod.MultimodalLSTM


class Stage02InferencePipeline:
    """
    Production-ready deployment pipeline wrapper for single-patient trajectory inference.
    """

    TABULAR_COLS = ["ctDNA_vaf_pct", "ct_tumor_vol_cm3", "biomarker_ng_ml", "wsi_atypia_score"]

    def __init__(
        self,
        model_path: str,
        scaler: StandardScaler,
        device: str = None
    ):
        """
        Args:
            model_path: Path to trained PyTorch model checkpoint (.pth).
            scaler: Fitted sklearn StandardScaler for tabular biomarkers.
            device: Computing device ('cpu' or 'cuda').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler

        script_dir = os.path.dirname(os.path.abspath(__file__))
        MultimodalLSTM = load_model_class(script_dir)

        self.model = MultimodalLSTM().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.transform_histo = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_ct = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _load_image(self, img_path: str, mode: str) -> Image.Image:
        """Safely loads an image or creates a fallback zero tensor Image if missing."""
        if os.path.exists(img_path):
            return Image.open(img_path).convert(mode)
        else:
            if mode == "RGB":
                return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                return Image.fromarray(np.zeros((224, 224), dtype=np.uint8))

    def _assign_risk_tier(self, prob: float) -> str:
        """Assigns clinical risk tier based on progression probability."""
        if prob < 0.35:
            return "Low Progression Risk"
        elif prob <= 0.70:
            return "Moderate Risk"
        else:
            return "High Progression Risk"

    def predict_patient_trajectory(
        self,
        patient_rows: Union[pd.DataFrame, List[Dict[str, Any]]],
        img_dir: str
    ) -> Dict[str, Any]:
        """
        Runs multi-modal trajectory inference on a single patient's 5 timepoints.

        Args:
            patient_rows: DataFrame or list of dicts for 1 patient (5 timepoints).
            img_dir: Root directory containing 2D pathology and CT scan images.

        Returns:
            Dict containing patient_id, cancer_type, timepoint probabilities, 
            overall trajectory risk, and clinical risk tier classification.
        """
        if isinstance(patient_rows, list):
            df_patient = pd.DataFrame(patient_rows)
        else:
            df_patient = patient_rows.copy()

        df_patient = df_patient.sort_values("sequence_step").reset_index(drop=True)

        if len(df_patient) != 5:
            raise ValueError(f"Inference requires exactly 5 timepoints for patient trajectory. Received {len(df_patient)}.")

        patient_id = df_patient["patient_id"].iloc[0]
        cancer_type = df_patient["cancer_type"].iloc[0]

        histo_tensors = []
        ct_tensors = []
        tabular_list = []
        timepoint_months = []

        for _, row in df_patient.iterrows():
            # 1. Pathology Tile Image
            h_path = os.path.join(img_dir, row["pathology_img_file"])
            if not os.path.exists(h_path):
                h_path = os.path.join(img_dir, cancer_type.lower(), os.path.basename(row["pathology_img_file"]))
            histo_img = self._load_image(h_path, mode="RGB")
            histo_tensors.append(self.transform_histo(histo_img))

            # 2. 2D CT Slice Image
            c_path = os.path.join(img_dir, row["ct_img_file"])
            if not os.path.exists(c_path):
                c_path = os.path.join(img_dir, cancer_type.lower(), os.path.basename(row["ct_img_file"]))
            ct_img = self._load_image(c_path, mode="L")
            ct_tensors.append(self.transform_ct(ct_img))

            # 3. Tabular Features
            raw_tab = row[self.TABULAR_COLS].values.astype(np.float32).reshape(1, -1)
            norm_tab = self.scaler.transform(raw_tab).squeeze(0)
            tabular_list.append(torch.tensor(norm_tab, dtype=torch.float32))

            timepoint_months.append(int(row["timepoint_month"]))

        # Stack into batch tensors of size [B=1, T=5, ...]
        h_seq = torch.stack(histo_tensors, dim=0).unsqueeze(0).to(self.device)  # [1, 5, 3, 224, 224]
        c_seq = torch.stack(ct_tensors, dim=0).unsqueeze(0).to(self.device)     # [1, 5, 1, 224, 224]
        tab_seq = torch.stack(tabular_list, dim=0).unsqueeze(0).to(self.device) # [1, 5, 4]

        # Model Inference
        with torch.no_grad():
            logits = self.model(h_seq, c_seq, tab_seq) # [1, 5]
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy() # [5]

        step_results = []
        for t in range(5):
            prob_t = float(probs[t])
            step_results.append({
                "step": t,
                "timepoint_month": timepoint_months[t],
                "progression_probability": round(prob_t, 4),
                "risk_tier": self._assign_risk_tier(prob_t)
            })

        final_prob = float(probs[-1])
        overall_risk_tier = self._assign_risk_tier(final_prob)

        return {
            "patient_id": patient_id,
            "cancer_type": cancer_type,
            "timepoint_predictions": step_results,
            "trajectory_final_progression_probability": round(final_prob, 4),
            "overall_clinical_risk_tier": overall_risk_tier
        }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load Data Loader helper to fit scaler
    loader_path = os.path.join(script_dir, "01_data_loader.py")
    spec_loader = importlib.util.spec_from_file_location("loader_mod", loader_path)
    loader_mod = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(loader_mod)

    csv_file = os.path.join(script_dir, "data", "stage02_lstm_8types_3000samples.csv")
    if not os.path.exists(csv_file):
        csv_file = os.path.join(script_dir, "stage02_lstm_8types_3000samples.csv")
    images_dir = os.path.join(script_dir, "2d_images_large")
    ckpt_path = os.path.join(script_dir, "best_stage02_multimodal_lstm.pth")

    print(f"--- Stage 02 Pipeline Integration & Inference Demo ---")
    df = pd.read_csv(csv_file)
    
    # Fit scaler on dataset
    scaler = StandardScaler()
    scaler.fit(df[Stage02InferencePipeline.TABULAR_COLS].values)

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint '{ckpt_path}' does not exist yet. Instantiating pipeline with un-trained architecture weights for demonstration.")
        # Save temporary checkpoint for demo if needed
        model_cls = load_model_class(script_dir)
        temp_model = model_cls()
        torch.save(temp_model.state_dict(), ckpt_path)

    # Initialize Inference Pipeline
    pipeline = Stage02InferencePipeline(model_path=ckpt_path, scaler=scaler)

    # Extract 5 rows for single patient 'PAT-0001'
    patient_id_demo = "PAT-0001"
    patient_df = df[df["patient_id"] == patient_id_demo]

    print(f"\nRunning Single-Patient Inference for {patient_id_demo}...")
    result = pipeline.predict_patient_trajectory(patient_df, images_dir)

    print("\n" + "=" * 60)
    print(f"SINGLE PATIENT INFERENCE REPORT: {result['patient_id']} ({result['cancer_type']})")
    print("=" * 60)
    print(f"Trajectory Final Progression Prob : {result['trajectory_final_progression_probability']:.4f}")
    print(f"Overall Clinical Risk Tier       : {result['overall_clinical_risk_tier']}")
    print("\nTimepoint Progression Breakdowns:")
    print(f"{'Step':<6} | {'Month':<8} | {'Probability':<14} | {'Risk Tier':<20}")
    print("-" * 55)
    for tp in result["timepoint_predictions"]:
        print(f"{tp['step']:<6} | M{tp['timepoint_month']:<7} | {tp['progression_probability']:<14.4f} | {tp['risk_tier']:<20}")
    print("=" * 60 + "\n")
