import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./qwen_coder_7b_merged"
awq_output_path = "./qwen_coder_7b_awq"

# AWQ Quantization Settings (INT4)
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

print("Loading merged model for AWQ quantization...")
model = AutoAWQForCausalLM.from_pretrained(
    model_path, 
    low_cpu_mem_usage=True, 
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Quantizing model (this will use GPU VRAM)...")
model.quantize(tokenizer, quant_config=quant_config)

print(f"Saving AWQ model to {awq_output_path}...")
model.save_quantized(awq_output_path)
tokenizer.save_pretrained(awq_output_path)

print("AWQ Quantization finished successfully!")
