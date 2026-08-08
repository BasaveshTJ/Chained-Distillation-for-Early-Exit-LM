import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "The capital of France is"

#######################################
# gpt2 117M
#######################################
print("########################################### GPT-2 ###########################################")
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=100, temperature=0.8, top_p=0.95,
        do_sample=True, pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("############################################################################")

outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


###########################################
# OPT-125M (~125M parameters)
###########################################
print("########################################### OPT-125M ###########################################")
model_name = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# ---------- Sampling generation ----------
with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=100, temperature=0.8, top_p=0.95,
        do_sample=True, pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("############################################################################")

# ---------- Greedy generation ----------
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


###########################################
# SmolLM2 135M (~135M parameters)
###########################################
print("########################################### SmolLM2 135M ###########################################")
# model_name = "HuggingFaceTB/SmolLM2-135M"
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# ---------- Sampling generation ----------
with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=100, temperature=0.8, top_p=0.95,
        do_sample=True, pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("############################################################################")

# ---------- Greedy generation ----------
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))