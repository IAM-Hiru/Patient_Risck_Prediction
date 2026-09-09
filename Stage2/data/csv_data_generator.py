import pandas as pd
import numpy as np

# Generates all 3,000 sequence rows across your 8 cancer types
cancers = ["BRCA", "LUAD", "LUSC", "COAD", "PRAD", "STAD", "SKCM", "BLCA"]
rows = []
global_idx = 0

for p in range(1, 601):
    patient_id = f"PAT-{p:04d}"
    cancer = cancers[(p - 1) % 8].lower()
    traj = np.random.choice([0.78, 1.25], p=[0.55, 0.45])
    
    for t in range(5):
        rows.append({
            "patient_id": patient_id,
            "cancer_type": cancer.upper(),
            "sequence_step": t,
            "timepoint_month": t * 3,
            "pathology_img_file": f"{cancer}/tile_{global_idx % 375:04d}.png",
            "ct_img_file": f"{cancer}/ct_slice_{global_idx % 375:04d}.png",
            "ctDNA_vaf_pct": round(max(0.01, 4.0 * (traj ** t)), 3),
            "ct_tumor_vol_cm3": round(max(0.1, 30.0 * (traj ** t)), 2),
            "biomarker_ng_ml": round(max(0.1, 20.0 * (traj ** t)), 2),
            "wsi_atypia_score": round(min(0.99, max(0.05, 0.4 * (traj ** (t * 0.5)))), 3),
            "label_progression": 1 if traj > 1.0 and t >= 2 else 0
        })
        global_idx += 1

pd.DataFrame(rows).to_csv("stage02_lstm_8types_3000samples.csv", index=False)