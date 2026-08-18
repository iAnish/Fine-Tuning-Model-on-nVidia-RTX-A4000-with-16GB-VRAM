# Fine-Tuning-Model-on-nVidia-RTX-A4000-with-16GB-VRAM
Fine-tuning a pre-quantized AWQ (Activation-aware Weight Quantization) model directly is generally not supported or recommended because AWQ weights are stored in integer formats (e.g., INT4), making backpropagation and gradient updates mathematically incompatible.

To achieve your goal on an NVIDIA RTX A4000 (16GB VRAM), the standard, industry-proven approach is to fine-tune the unquantized 16-bit model (Qwen/Qwen2.5-Coder-7B-Instruct) using QLoRA (4-bit BitsAndBytes quantization), and then optional re-quantization to AWQ after fine-tuning.

# Prerequisites & Dependecies
Install the required libraries in your Python environment:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes trl

# Screenshot of the fine tuning
<img width="1041" height="533" alt="image" src="https://github.com/user-attachments/assets/0fe13084-cf28-47d4-8816-02e2415d65b0" />


<img width="1124" height="628" alt="image" src="https://github.com/user-attachments/assets/8170b6a4-fdad-4711-a8f2-e73d7bb6055f" />
