import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
# prompt = "The capital of France is "
prompt = [
   {
    "content": "Hi there",
    "role": "user"
  },
  {
    "content": "Hello! How can I help you today?",
    "role": "assistant"
  },
  {
    "content": "I'm looking for a beach resort for my next vacation. Can you recommend some popular ones?",
    "role": "user"
  } ]

#######################################
# gpt2 117M
#######################################
# print("########################################### GPT-2 ###########################################")
# model_name = "gpt2"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
# model.eval()

# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.8, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# print("############################################################################")

# outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# ###########################################
# # OPT-125M (~125M parameters)
# ###########################################
# print("########################################### OPT-125M ###########################################")
# model_name = "facebook/opt-125m"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
# model.eval()

# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# # ---------- Sampling generation ----------
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.8, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# print("############################################################################")

# # ---------- Greedy generation ----------
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


###########################################
# SmolLM2 135M (~135M parameters)
###########################################
print("########################################### SmolLM2 135M ###########################################")
# model_name = "HuggingFaceTB/SmolLM2-135M"
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model_path = "./smollm2-135m-finetuned/checkpoint-670"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_without_EE = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
model_without_EE.eval()
from model.SmolLM2EarlyExitForCausalLM import SmolLM2EarlyExitForCausalLM
model_with_EE = SmolLM2EarlyExitForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
model_with_EE.eval()


# inputs = tokenizer(prompt, return_tensors="pt").to(device)
inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

print("\n################################ without EE ##################################")
# ---------- Sampling generation ----------
with torch.no_grad():
    outputs = model_without_EE.generate(**inputs, max_new_tokens=100, temperature=0.8, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("############################################################################")

# ---------- Greedy generation ----------
with torch.no_grad():
    outputs = model_without_EE.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


print("\n################################ with EE ##################################")
# ---------- Sampling generation ----------
with torch.no_grad():
    outputs = model_with_EE.generate(**inputs, max_new_tokens=100, temperature=0.8, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("############################################################################")

# ---------- Greedy generation ----------
with torch.no_grad():
    outputs = model_with_EE.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))