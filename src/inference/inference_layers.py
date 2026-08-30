import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
# prompt = "berlin and paris are "
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
  },
   {
    "content": "Some",
    "role": "assistant"
  }
   ]

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
        print(f"{rank + 1}. {repr(token):15} {top_probs[0, rank].item():.6f}")

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
        print(f"{rank + 1}. {repr(token):15} {top_probs[0, rank].item():.6f}")

# Last layer prediction
h = model.model.decoder.final_layer_norm(hidden_states[-1][:, -1, :])
probs = torch.softmax(model.lm_head(h), dim=-1)
top_probs, top_ids = torch.topk(probs, k=5)
print(f"\nLayer {len(hidden_states) - 1}")
for rank in range(5):
    token = tokenizer.decode([top_ids[0, rank].item()])
    print(f"{rank+1}. {repr(token):15} {top_probs[0, rank].item():.6f}")


# ======================================================
# SmolLM2 135M
# ======================================================
print("\n########################################### SmolLM2 135M ###########################################")
base_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
finetuned_model_path = "./smollm2-135m-finetuned/checkpoint-670"
top_k = 5

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model_without_EE = AutoModelForCausalLM.from_pretrained(base_model_name, output_hidden_states=True).to(device)
from model.SmolLM2EarlyExitForCausalLM import SmolLM2EarlyExitForCausalLM
model_with_EE = SmolLM2EarlyExitForCausalLM.from_pretrained(finetuned_model_path, output_hidden_states=True).to(device)

model_without_EE.eval()
model_with_EE.eval()


def _topk_rows(logits, tokenizer, k=5):
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k)
    rows = []
    for rank in range(k):
        token = tokenizer.decode([top_ids[rank].item()]).replace("\n", "\\n")
        rows.append(f"{rank + 1}:{repr(token)} {top_probs[rank].item():.4f}")
    return rows


def _layer_logits_last_token(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states
    num_layers = model.config.num_hidden_layers
    layer_logits = []

    for layer_idx in range(1, num_layers + 1):
        h = hidden_states[layer_idx][:, -1, :]
        if layer_idx < num_layers:
            h = model.model.norm(h)
        layer_logits.append(model.lm_head(h).squeeze(0))

    return layer_logits


inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

base_logits_by_layer = _layer_logits_last_token(model_without_EE, inputs)
ft_logits_by_layer = _layer_logits_last_token(model_with_EE, inputs)

print("Number of layers:", len(base_logits_by_layer))
print("========== Side-by-side per-layer logits (last token) ==========")
print("Columns: Base model top-k || Fine-tuned early-exit model top-k")

for layer_idx, (base_logits, ft_logits) in enumerate(zip(base_logits_by_layer, ft_logits_by_layer), start=1):
    cos_sim = F.cosine_similarity(base_logits.unsqueeze(0), ft_logits.unsqueeze(0)).item()
    base_rows = _topk_rows(base_logits, tokenizer, top_k)
    ft_rows = _topk_rows(ft_logits, tokenizer, top_k)

    print(f"\nLayer {layer_idx:02d} | cosine={cos_sim:.4f}")
    print(f"{'Base:':<32}FT:")
    for base_row, ft_row in zip(base_rows, ft_rows):
        print(f"{base_row:<32}{ft_row}")
