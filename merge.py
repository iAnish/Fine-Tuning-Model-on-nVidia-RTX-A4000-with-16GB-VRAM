import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter_path = "./qwen_coder_lora_final"
merged_output_path = "./qwen_coder_7b_merged"

print("Loading base model in FP16...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu",  # Load on CPU first to save GPU VRAM for quantization
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {merged_output_path}...")
model.save_pretrained(merged_output_path, safe_serialization=True)
tokenizer.save_pretrained(merged_output_path)

print("Merge complete!")
