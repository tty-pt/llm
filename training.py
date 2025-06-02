from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 1. Modelo base
local_model_path = snapshot_download("mistralai/Mistral-7B-Instruct-v0.3")

tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(local_model_path)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    config=config,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
)

for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

# 2. PEFT config
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files="btestf.jsonl", split="train")
print(f"Dataset size: {len(dataset)}")
print(dataset[0])
dataset = dataset.rename_column("prompt", "text")
# dataset = dataset.select(range(50))

# 3. Tokenização
def tokenize_fn(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    out["labels"] = out["input_ids"].copy()
    return out
# def tokenize_fn(example):
#     return tokenizer(
#         example["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=512
#     )

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# 4. Args
training_args = TrainingArguments(
    output_dir="./mistral-lora",
    per_device_train_batch_size=1,
    warmup_steps=10,
    max_steps=2500,
    logging_steps=50,
    save_steps=500,
    gradient_accumulation_steps=8,
    fp16=False,
    bf16=False,
    no_cuda=True,
    report_to="none"
)

training_args.max_steps = 2500

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

