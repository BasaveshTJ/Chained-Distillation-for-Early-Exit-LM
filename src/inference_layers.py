import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "The capital of France is "

# ======================================================
# GPT-2
# ======================================================
print("########################################### GPT-2 ###########################################")
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

hidden_states = outputs.hidden_states

print("Number of hidden states:", len(hidden_states))
print("Expected: 13 (embedding + 12 blocks)")

# Logit lens on Shallow layers
print("\n========== Shallow Layers predictions ==========")

for layer_idx, hidden in enumerate(hidden_states[1:-1]):
    h = model.transformer.ln_f(hidden[:, -1, :])
    logits = model.lm_head(h)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=5)
    print(f"\nLayer {layer_idx + 1}")
    for rank in range(5):
        token = tokenizer.decode([top_ids[0, rank].item()])
        print(f"{rank+1}. {repr(token):15} {top_probs[0, rank].item():.6f}")

# last layer
h = hidden_states[-1][:, -1, :]
probs = torch.softmax(model.lm_head(h), dim=-1)
top_probs, top_ids = torch.topk(probs, k=5)
print(f"\nLayer {len(hidden_states) - 1}")
for rank in range(5):
    token = tokenizer.decode([top_ids[0, rank].item()])
    print(f"{rank+1}. {repr(token):15} {top_probs[0, rank].item():.6f}")


# ======================================================
# OPT-125M
# ======================================================
print("\n########################################### OPT-125M ###########################################")
model_name = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

hidden_states = outputs.hidden_states

print("Number of hidden states:", len(hidden_states))

# Logit lens on Shallow layers
print("========== Shallow Layers predictions ==========")

for layer_idx, hidden in enumerate(hidden_states[1:-1]):
    h = model.model.decoder.final_layer_norm(hidden[:, -1, :])
    probs = torch.softmax(model.lm_head(h), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=5)
    print(f"\nLayer {layer_idx + 1}")
    for rank in range(5):
        token = tokenizer.decode([top_ids[0, rank].item()])
        print(f"{rank+1}. {repr(token):15} {top_probs[0, rank].item():.6f}")

# Last layer prediction
h = model.model.decoder.final_layer_norm(hidden_states[-1][:, -1, :])
probs = torch.softmax(model.lm_head(h), dim=-1)
top_probs, top_ids = torch.topk(probs, k=5)
print(f"\nLayer {len(hidden_states)-1}")
for rank in range(5):
    token = tokenizer.decode([top_ids[0, rank].item()])
    print(f"{rank+1}. {repr(token):15} {top_probs[0, rank].item():.6f}")


# ======================================================
# SmolLM2 135M
# ======================================================
print("\n########################################### SmolLM2 135M ###########################################")
model_name = "HuggingFaceTB/SmolLM2-135M"
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
model.eval()
messages_think = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

hidden_states = outputs.hidden_states

print("Number of hidden states:", len(hidden_states))

# Logit lens on Shallow layers
print("========== Shallow Layers predictions ==========")

for layer_idx, hidden in enumerate(hidden_states[1:-1]):
    h = model.model.norm(hidden[:, -1, :])
    probs = torch.softmax(model.lm_head(h), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=5)
    print(f"\nLayer {layer_idx}")

    for rank in range(5):
        token = tokenizer.decode([top_ids[0, rank].item()])
        print(f"{rank + 1}. {repr(token):15} {top_probs[0, rank].item():.6f}")

# Last layer prediction
h = model.model.norm(hidden_states[-1][:, -1, :])
probs = torch.softmax(model.lm_head(h), dim=-1)
top_probs, top_ids = torch.topk(probs, k=5)
print(f"\nLayer {len(hidden_states)-1}")
for rank in range(5):
    token = tokenizer.decode([top_ids[0, rank].item()])
    print(f"{rank+1}. {repr(token):15} {top_probs[0, rank].item():.6f}")
