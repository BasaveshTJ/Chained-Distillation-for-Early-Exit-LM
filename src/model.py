import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# gpt2 117M
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\n################ Model Architecture ################")
print("model arch:", model)

print("\n################ Model Config: ################")
print( model.config)


model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\n################ Model Architecture ################")
print("model arch:", model)

print("\n################ Model Config: ################")
print(model.config)