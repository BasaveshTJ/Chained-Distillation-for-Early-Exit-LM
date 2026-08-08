import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# gpt2 
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\n################ Model Architecture ################")
print("model arch:", model)

print("\n################ Model Config ################")
print(model.config)

print("\n total parameters:", sum(p.numel() for p in model.parameters()))  # gpt2 : 124,439,808

# OPT 125M
model_name = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\n################ Model Architecture ################")
print("model arch:", model)

print("\n################ Model Config ################")
print(model.config)

print("\n total parameters:", sum(p.numel() for p in model.parameters()))  # opt 125m : 125,239,296


# SmolLM2 135M
# model_name = "HuggingFaceTB/SmolLM2-135M"
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\n################ Model Architecture ################")
print("model arch:", model)

print("\n################ Model Config ################")
print(model.config)


print("\n total parameters:", sum(p.numel() for p in model.parameters()))  # SmolLM2 135M : 134,515,008