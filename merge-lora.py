from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "mistralai/Mistral-7B-Instruct-v0.3"
LORA = "mistral-lora/checkpoint-1000"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model_base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="cpu")

model_lora = PeftModel.from_pretrained(model_base, LORA)
merged = model_lora.merge_and_unload()

merged.save_pretrained("mistral-merged")
tokenizer.save_pretrained("mistral-merged")
