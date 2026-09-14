import importlib.util
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def append_token(model_inputs, token_id):
    model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], token_id.unsqueeze(-1)], dim=-1)
    if "attention_mask" in model_inputs:
        ones = torch.ones_like(token_id, dtype=model_inputs["attention_mask"].dtype).unsqueeze(-1)
        model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], ones], dim=-1)


def topk_tokens(tokenizer, probs, k=5):
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    return [
        (top_ids[0, i].item(), top_probs[0, i].item(), tokenizer.decode([top_ids[0, i].item()], skip_special_tokens=False))
        for i in range(k)
    ]


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "HuggingFaceTB/SmolLM2-135M"
deepadapt_ckpt = "./smollm2-135m-deepadapt/checkpoint-201"
max_new_tokens = 30
confidence_threshold = 0.5

prompt = """User: Hi there
Assistant: Hello! How can I help you today?
User: I'm looking for a beach resort for my next vacation. Can you recommend some popular ones?
Assistant:"""

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained( model_name,torch_dtype=torch.float16 if device == "cuda" else torch.float32,).to(device)
base_model.eval()
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nBase manual greedy (token + confidence)")
base_inputs = {k: v.clone() for k, v in inputs.items()}
base_generated = []

with torch.no_grad():
    for _ in range(max_new_tokens):
        logits = base_model(**base_inputs, output_attentions=False, output_hidden_states=False).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.argmax(probs, dim=-1)
        next_conf = torch.gather(probs, 1, next_id.unsqueeze(-1)).squeeze(-1)
        base_generated.append((tokenizer.decode([next_id.item()], skip_special_tokens=False), next_conf.item()))
        append_token(base_inputs, next_id)
        if next_id.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(base_inputs["input_ids"][0], skip_special_tokens=True))
for token_text, confidence in base_generated:
    print(f"confidence={confidence:.4f} | token={repr(token_text)}")

current_dir = os.path.dirname(__file__)
model_file = os.path.join(os.path.abspath(os.path.join(current_dir, "..")), "model", "SmolLM2DeepAdaptForCausalLM.py")
spec = importlib.util.spec_from_file_location("deepadapt_model", model_file)
deepadapt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepadapt_module)
SmolLM2DeepAdaptForCausalLM = deepadapt_module.SmolLM2DeepAdaptForCausalLM


deepadapt_model = SmolLM2DeepAdaptForCausalLM.from_pretrained(deepadapt_ckpt,torch_dtype=torch.float16 if device == "cuda" else torch.float32,).to(device)
deepadapt_model.set_confidence_threshold(confidence_threshold)
deepadapt_model.eval()

print("\nDeepAdapt greedy (exit 30/31 gating, top5 vs top5)")
deep_inputs = {k: v.clone() for k, v in inputs.items()}
deep_generated = []

with torch.no_grad():
    for step in range(max_new_tokens):
        outputs = deepadapt_model(**deep_inputs, output_attentions=False, output_hidden_states=False)
        probs_30 = torch.softmax(outputs.base_logits[:, -1, :], dim=-1)
        probs_31 = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        conf_30, token_30 = torch.max(probs_30, dim=-1)
        conf_31, token_31 = torch.max(probs_31, dim=-1)

        use_30 = conf_30.item() >= confidence_threshold
        next_id = token_30 if use_30 else token_31
        deep_generated.append( { "step": step + 1, "exit": 30 if use_30 else 31,
                                "top5_30": topk_tokens(tokenizer, probs_30, k=5),
                                "top5_31": topk_tokens(tokenizer, probs_31, k=5), } )

        append_token(deep_inputs, next_id)
        if next_id.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(deep_inputs["input_ids"][0], skip_special_tokens=True))
for item in deep_generated:
    print(f"\nStep {item['step']:>2} | used_exit={item['exit']}")
    print("  exit30_top5                           || exit31_top5")
    for i in range(5):
        _, p30, t30 = item["top5_30"][i]
        _, p31, t31 = item["top5_31"][i]
        print(f"  p={p30:.4f} | token={repr(t30):<18} || p={p31:.4f} | token={repr(t31)}")

count_30 = sum(1 for item in deep_generated if item["exit"] == 30)
count_31 = len(deep_generated) - count_30
print(f"\nTotal tokens: {len(deep_generated)}")
print(f"Exit30: {count_30}")
print(f"Exit31: {count_31}")