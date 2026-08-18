import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. Load Data
def load_text_files(folder_path="./data"):
    texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append({"text": content})
    return Dataset.from_list(texts)

dataset = load_text_files("./data")

# 2. Model & Tokenizer Config
model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# 3. LoRA Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./qwen_coder_lora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    max_steps=100,  # Adjust based on dataset size
    learning_rate=2e-4,
    fp16=False,
    bf16=True,  # RTX A4000 supports Ampere bfloat16
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_8bit",
)


# Update your training arguments to use SFTConfig instead of TrainingArguments
sft_config = SFTConfig(
    output_dir="./qwen_coder_lora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    max_steps=100,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    max_length=2048,  # Replaces max_seq_length in newer TRL versions
)


# 5. SFT Trainer Initialization
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_config,
#     dataset_text_field="text",
#     max_seq_length=2048,
#     tokenizer=tokenizer,
#     args=training_args,
# )

# 6. Train & Save
trainer.train()
trainer.model.save_pretrained("./qwen_coder_lora_final")
tokenizer.save_pretrained("./qwen_coder_lora_final")
