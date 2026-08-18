# Fine-Tuning-Model-on-nVidia-RTX-A4000-with-16GB-VRAM
# Qwen2.5-Coder Fine-Tuning & Quantization Pipeline

A lightweight repository for fine-tuning `Qwen/Qwen2.5-Coder-7B-Instruct` on custom dataset files using QLoRA, merging adapters, and quantizing to AWQ format on consumer hardware (e.g., NVIDIA RTX A4000 16GB).

---

## ⚡ Quick Start

### 1. Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes trl autoawq

```

### 2. Dataset Setup

Place your raw `.txt` files into a folder named `./data/`:

```
.
├── data/
│   ├── sample1.txt
│   └── sample2.txt
├── train.py
├── merge.py
└── quantize.py

```

---

## 🚀 Execution Workflow

### Step 1: Fine-Tune with QLoRA

Run SFT training using bitsandbytes 4-bit quantization and TRL:

```bash
python train.py

```

### Step 2: Merge LoRA Adapters

Merge trained LoRA weights back into the FP16 base model:

```bash
python merge.py

```

### Step 3: Quantize to 4-bit AWQ

Export the merged model to high-performance AWQ format for deployment:

```bash
python quantize.py

```

---

## ⚙️ Key Configuration

| Parameter | Default Value | Notes |
| --- | --- | --- |
| **Base Model** | `Qwen/Qwen2.5-Coder-7B-Instruct` | Unquantized FP16 |
| **Context Length** | `2048` | Reduce if encountering OOM |
| **Quantization** | `4-bit (NF4 / AWQ)` | Fits within 16GB VRAM |

# Screenshot of the fine tuning
<img width="1041" height="533" alt="image" src="https://github.com/user-attachments/assets/0fe13084-cf28-47d4-8816-02e2415d65b0" />


<img width="1124" height="628" alt="image" src="https://github.com/user-attachments/assets/8170b6a4-fdad-4711-a8f2-e73d7bb6055f" />
