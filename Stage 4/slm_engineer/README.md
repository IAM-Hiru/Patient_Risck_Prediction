# Stage 04 SLM Fine-Tuning & ML Engineer

## Overview
This module conducts 4-bit QLoRA instruction fine-tuning on a 3-Billion parameter Small Language Model (`Qwen/Qwen2.5-3B-Instruct`) to generate strict 2-sentence clinical summaries for tumor boards with zero hallucinations and accurate triage classification.

---

## Directory Structure
```
Stage 4/slm_engineer/
├── models/
│   └── stage04_slm_qlora/
│       ├── adapter_config.json          # LoRA PEFT target module configuration
│       ├── adapter_model.safetensors    # Quantized trained adapter weights (61.1 MB)
│       ├── special_tokens_map.json      # ChatML delimiters (<|im_start|>, <|im_end|>)
│       ├── tokenizer_config.json        # Fast tokenizer settings & max_length (512)
│       └── tokenizer.json               # Fast tokenizer vocab & padding specs
├── outputs/
│   └── stage04_training_report.json     # Execution report, loss metrics, and VRAM profile
├── src/
│   └── stage04_train_slm.py             # QLoRA fine-tuning implementation
├── stage04_slm_pipeline.py              # Orchestrator runner
└── README.md
```

---

## Fine-Tuning Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Base Model Backbone** | `Qwen/Qwen2.5-3B-Instruct` | 3.09B parameter instruction-tuned causal LLM |
| **Quantization Precision** | `nf4` (4-bit NormalFloat) | Double quantization with 8-bit page optimization |
| **LoRA Rank ($r$)** | `16` | Rank dimension for adapter matrices |
| **LoRA Alpha ($\alpha$)** | `32` | Scaling factor ($\alpha / r = 2.0$) |
| **LoRA Dropout** | `0.05` | Regularization dropout rate |
| **Target Projections** | All 7 Linear Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Trainable Parameters** | `20,185,088` (0.65%) | Total parameters: 3,090,000,000 |
| **Learning Rate** | `2e-4` | Cosine annealing schedule |
| **Batch Size (Effective)** | `16` | Per-device batch 8 $\times$ Gradient accumulation 2 |
| **Max Sequence Length** | `512 tokens` | Formatted under ChatML schema |
| **Epochs** | `3` | Convergence reached at final loss 0.1245 |

---

## Execution
Run from root:
```powershell
python "Stage 4/slm_engineer/stage04_slm_pipeline.py"
```
Or directly from `src`:
```powershell
python "Stage 4/slm_engineer/src/stage04_train_slm.py"
```
