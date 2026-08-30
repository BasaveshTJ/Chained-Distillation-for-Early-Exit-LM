import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# prompt = [
#       {
#     "content": "Hey!",
#     "role": "user"
#   },
#   {
#     "content": "Hello! How can I help you today?",
#     "role": "assistant"
#   },
#   {
#     "content": "I'm planning a trip to Paris. What are some popular tourist attractions?",
#     "role": "user"
#   }
#     ]

prompt = [
      {
    "content": "Provide a concise, objective summary of the input text in up to three sentences, focusing on key actions and intentions without using second or third person pronouns.",
    "role": "system"
  },
  {
    "content": "By . Emily Crane . The competition watchdog is taking Virgin Australia and Jetstar to court, accusing them of slugging customers with hidden fees when buying cheap airfares online. The airlines are alleged to have engaged in 'drip pricing', where a headline price is advertised but extra fees and charges are introduced during the booking process. The Australian Competition and Consumer Commission (ACCC) has launched separate Federal Court proceedings against Virgin and Jetstar, with the airlines facing hefty fines and orders to change their advertising practices. The competition watchdog is taking Virgin Australia and Jetstar to court, accusing them of slugging customers with hidden fees when buying cheap airfares online . Jetstar said it will defend the allegations, while Virgin is reviewing the ACCC's proceedings and considering its options. Jetstar is alleged to have failed to inform customers they would be charged a booking fee of $8.50 per passenger when they paid online with a credit card. Virgin allegedly includes a booking fee of $7.70 for those paying with a credit card, debit card or via PayPal. The airlines are alleged to have engaged in 'drip pricing', where a headline price is advertised but extra fees and charges are introduced during the booking process . 'The ACCC alleges that these fees applied to the substantial majority of online bookings and should have been disclosed upfront and prominently with or within headline prices,' the watchdog said. Jetstar said it offers airline seats at the lowest price, and then optional extras - including fees for a particular way of booking. 'The booking and service fee is clearly disclosed and the total price that people pay is shown before they finalise their purchase,' a Jetstar spokesperson said. The Australian Competition and Consumer Commission chairman Rod Sims said the watchdog was investigating similar 'drip pricing' behaviour by businesses in other industries after Federal Court proceedings were launched against Virgin and Jetstar . 'Our customers have the option to choose one of four fee-free payment methods and that's how a large number of them book.' Virgin said all Australian airlines have long charged separate booking and service fees and its customers are also offered fee-free payment options. More... Could re-routing flights reduce climate change? Avoiding certain areas and weather conditions may help prevent global warming . Passenger gets stuck in toilet on 15 hour flight after getting finger lodged in lavatory bin . ACCC chairman Rod Sims said the watchdog was investigating similar 'drip pricing' behaviour by businesses in other industries. The court proceedings against the airlines will begin in August. Jetstar said it will defend the allegations, while Virgin is reviewing the ACCC's proceedings and considering its options .",
    "role": "user"
  }
]

# ======================================================
# SmolLM2 135M
# ======================================================
print("\n########################################### SmolLM2 135M ###########################################")
base_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
finetuned_model_path = "./smollm2-135m-finetuned/checkpoint-670"
# finetuned_model_path = "./smollm2-135m-finetuned_lr_2e5/checkpoint-2010"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model_without_EE = AutoModelForCausalLM.from_pretrained(base_model_name, output_hidden_states=True).to(device)
from model.SmolLM2EarlyExitForCausalLM import SmolLM2EarlyExitForCausalLM
model_with_EE = SmolLM2EarlyExitForCausalLM.from_pretrained(finetuned_model_path, output_hidden_states=True).to(device)

model_without_EE.eval()
model_with_EE.eval()


def early_exit_inference_per_token(model, inputs, exit_layers, threshold=0.5):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states
    num_layers = model.config.num_hidden_layers
    
    # check the token probabilities at the given exit layers, if the probability of the most likely token exceeds a threshold, exit early
    for layer_idx in exit_layers:
        h = hidden_states[layer_idx][:, -1, :]
        if layer_idx < num_layers:
            h = model.model.norm(h)
        logits = model.lm_head(h).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_id = torch.max(probs, dim=-1)
        # print(f"Layer {layer_idx}: top token id={top_id.item()}, prob={top_prob.item():.4f}")
        if top_prob.item() >= threshold:
            # print(f"Early exit at layer {layer_idx}")
            return top_id.item(), top_prob.item(), layer_idx

    # If no early exit, return the final layer's top token
    h = hidden_states[-1][:, -1, :]
    if num_layers > 0:
        h = model.model.norm(h)
    logits = model.lm_head(h).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    top_prob, top_id = torch.max(probs, dim=-1)
    # print(f"Final layer: top token id={top_id.item()}, prob={top_prob.item():.4f}")
    return top_id.item(), top_prob.item(), num_layers

def run_inference_with_early_exit(model, inputs, exit_layers, threshold=0.5, max_new_tokens=10):
    generated_ids = inputs["input_ids"]
    exit_layer_counts = {layer: [] for layer in exit_layers}
    for _ in range(max_new_tokens):
        next_token_id, next_token_prob, exit_layer = early_exit_inference_per_token(model, {"input_ids": generated_ids}, exit_layers, threshold)
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=generated_ids.device)], dim=-1)
        exit_layer_counts[exit_layer].append(next_token_id)
        if next_token_id == tokenizer.eos_token_id:
            break
    return generated_ids, exit_layer_counts

inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
max_new_tokens = 30
threshold = 0.6
exit_layers = [10, 20, 30]
with torch.no_grad():
    outputs = model_without_EE.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print("Without Early Exit:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\nWith Early Exit: base model")
generated_ids, exit_layer_counts = run_inference_with_early_exit(model_without_EE, inputs, exit_layers=exit_layers, threshold=threshold, max_new_tokens=max_new_tokens)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
print("Exit layer counts:", exit_layer_counts)

print("\nWith Early Exit: fine tuned model")
generated_ids, exit_layer_counts = run_inference_with_early_exit(model_with_EE, inputs, exit_layers=exit_layers, threshold=threshold, max_new_tokens=max_new_tokens)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
print("Exit layer Token ids:")
for layer, token_ids in exit_layer_counts.items():
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    print(f"Layer {layer}: {tokens} , Total tokens exited: {len(token_ids)}")