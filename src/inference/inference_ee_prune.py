import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
FINETUNED_MODEL_PATH = "./smollm2-135m-finetuned/checkpoint-670"
EXIT_LAYERS = [10, 20, 30]
THRESHOLD = 0.6
MAX_NEW_TOKENS = 30
DO_SAMPLE = False


class VariableDepthSmolLM2:
    """
    Implements variable-depth inference for SmolLM2.

    Prompt tokens:
        Always execute layers 1..30.

    Generated tokens:
        Execute until they confidently exit at layer 10,
        20, or otherwise continue to layer 30.

        If token A exits at layer 10:
            A K/V exists at layers 1..10.
            A K/V does NOT exist for attention at layers 11..30.

        If token B exits at layer 20:
            B K/V exists at layers 1..20.
            B does NOT exist in attention at layers 21..30.

    Therefore every transformer layer has its own KV cache.
    """

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.transformer = model.model
        self.layers = self.transformer.layers
        self.lm_head = model.lm_head
        self.final_norm = self.transformer.norm
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.hidden_size // self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        assert self.num_layers == 30, f"Expected 30 layers, got {self.num_layers}"

        print("=" * 80, "Variable-depth SmolLM2", "=" * 80, f"Layers              : {self.num_layers}", f"Hidden size         : {self.hidden_size}", f"Attention heads     : {self.num_attention_heads}", f"KV heads            : {self.num_key_value_heads}", f"Head dimension      : {self.head_dim}", f"Exit layers         : {EXIT_LAYERS}", f"Threshold           : {THRESHOLD}", "=" * 80, sep="\n")

    def rotary_embedding(self, hidden_states, position_ids):
        rotary_emb = self.transformer.rotary_emb
        try:
            # Modern Transformers
            cos, sin = rotary_emb( hidden_states, position_ids=position_ids)
            return cos, sin
        except TypeError:
            max_position = int(position_ids.max().item()) + 1
            cos, sin = rotary_emb(hidden_states, seq_len=max_position )
            if cos.dim() == 3:
                cos = cos[:, position_ids[0]]
                sin = sin[:, position_ids[0]]
            return cos, sin

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rope(self, q, k, position_ids):
        dummy = torch.empty(1, position_ids.shape[-1], self.hidden_size, device=q.device, dtype=q.dtype)

        cos, sin = self.rotary_embedding(dummy, position_ids)

        if cos.dim() == 2:
            cos = cos.unsqueeze(0)

        if sin.dim() == 2:
            sin = sin.unsqueeze(0)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)

        return q, k

    def repeat_kv(self, x):
        if self.num_key_value_groups == 1:
            return x

        B, KVH, S, D = x.shape
        x = x[:, :, None, :, :]
        x = x.expand(B, KVH, self.num_key_value_groups, S, D)
        x = x.reshape(B, self.num_attention_heads, S, D)
        return x

    def attention(self, layer, hidden_states, position_ids, kv_cache):
        attn = layer.self_attn
        B, Q, _ = hidden_states.shape
        query_states = attn.q_proj(hidden_states)
        query_states = query_states.view(B, Q, self.num_attention_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = attn.k_proj(hidden_states)
        key_states = key_states.view(B, Q, self.num_key_value_heads, self.head_dim)
        key_states = key_states.transpose(1, 2)
        value_states = attn.v_proj(hidden_states)
        value_states = value_states.view(B, Q, self.num_key_value_heads, self.head_dim)
        value_states = value_states.transpose(1, 2)
        query_states, key_states = self.apply_rope(query_states, key_states, position_ids)

        # Existing cache contains ONLY tokens that have
        if kv_cache is not None:
            past_k, past_v = kv_cache
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        new_cache = (key_states, value_states)
        key_for_attention = self.repeat_kv(key_states)
        value_for_attention = self.repeat_kv(value_states)

        attn_weights = torch.matmul(query_states, key_for_attention.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        kv_len = key_states.shape[-2]

        if Q > 1:

            q_positions = position_ids
            k_positions = torch.arange(kv_len, device=hidden_states.device)
            causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            causal_mask = causal_mask.to(dtype=torch.bool)
            causal_mask = causal_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(~causal_mask, torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_for_attention)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, Q, self.num_attention_heads * self.head_dim)
        attn_output = attn.o_proj(attn_output)
        return attn_output, new_cache

    def decoder_layer(self, layer_idx, hidden_states, position_ids, kv_cache):
        """
        Manually execute one Llama decoder layer.
        """

        layer = self.layers[layer_idx]
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, new_cache = self.attention(layer, hidden_states, position_ids, kv_cache)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp = layer.mlp
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_cache

    def exit_norm(self, hidden_states, layer_number):
        if layer_number == 10:
            if not hasattr(self.model, "exit_norm_10"):
                raise RuntimeError("Model does not contain exit_norm_10")
            return self.model.exit_norm_10(hidden_states)
        elif layer_number == 20:
            if not hasattr(self.model, "exit_norm_20"):
                raise RuntimeError("Model does not contain exit_norm_20")
            return self.model.exit_norm_20(hidden_states)
        elif layer_number == 30:
            return self.final_norm(hidden_states)
        else:
            raise ValueError(f"Unsupported exit layer: {layer_number}")

    def predict_from_hidden(self, hidden_states, layer_number):
        normalized = self.exit_norm(hidden_states, layer_number)
        logits = self.lm_head(normalized)
        probs = F.softmax(logits, dim=-1)
        confidence, token_ids = torch.max(probs, dim=-1)
        return (logits, probs,confidence, token_ids )

    @torch.no_grad()
    def select_first_token_from_model_forward(self, input_ids, verbose=True):
        """
        Match first-token early-exit selection used in inference_ee.py:
        run a native forward pass over the prompt and inspect exit layers in order.
        """

        outputs = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        for layer_number in EXIT_LAYERS:
            h = hidden_states[layer_number][:, -1, :]
            if layer_number < self.num_layers:
                h = self.final_norm(h)
            logits = self.lm_head(h)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted_token = torch.max(probs, dim=-1)

            confidence_value = confidence.item()
            predicted_id = predicted_token.item()

            if verbose:
                print(f"  first-token layer {layer_number:2d} candidate={predicted_id:5d} confidence={confidence_value:.4f} threshold={THRESHOLD:.4f}")

            if confidence_value >= THRESHOLD:
                return predicted_id, layer_number, confidence_value

        final_hidden = hidden_states[-1][:, -1, :]
        final_hidden = self.final_norm(final_hidden)
        final_logits = self.lm_head(final_hidden)
        final_probs = F.softmax(final_logits, dim=-1)
        final_confidence, final_token = torch.max(final_probs, dim=-1)
        return final_token.item(), self.num_layers, final_confidence.item()

    @torch.no_grad()
    def prefill_prompt(self, input_ids, verbose=True):
        """
        Process the entire prompt. Prompt tokens NEVER exit.
        Therefore every prompt token is propagated through all 30 layers.
        """

        outputs = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
        kv_cache = outputs.past_key_values
        return kv_cache, None, None, None, None

    @torch.no_grad()
    def generate_one_token(self, token_id, position, kv_cache, verbose=True):

        # Process ONE newly generated token.
        # The token starts at layer 1.
        # At each layer:
        #     token participates in attention
        #     token's K/V is added to that layer cache
        # At an exit layer:
        #     calculate confidence
        #     if confident:
        #         stop processing this token

        outputs = self.model(
            input_ids=token_id,
            past_key_values=kv_cache,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

        new_kv_cache = outputs.past_key_values
        hidden_states = outputs.hidden_states
        predicted_id = None
        exited_at = None
        exit_confidence = None

        for layer_number in EXIT_LAYERS:
            h = hidden_states[layer_number][:, -1:, :]
            logits, probs, confidence, predicted_token = self.predict_from_hidden(h, layer_number)
            confidence_value = confidence[0, 0].item()
            candidate_id = predicted_token[0, 0].item()

            if verbose:
                print(f"      layer {layer_number:2d} candidate={candidate_id:5d} confidence={confidence_value:.4f} threshold={THRESHOLD:.4f}")

            if layer_number < self.num_layers and confidence_value >= THRESHOLD:
                predicted_id = candidate_id
                exited_at = layer_number
                exit_confidence = confidence_value
                break

        if predicted_id is None:
            h = hidden_states[-1][:, -1:, :]
            final_logits, final_probs, confidence, predicted_token = self.predict_from_hidden(h, self.num_layers)
            predicted_id = predicted_token[0, 0].item()
            exit_confidence = confidence[0, 0].item()
            exited_at = self.num_layers
            if verbose:
                print(f"      layer 30 candidate={predicted_id:5d} confidence={exit_confidence:.4f} FINAL")

        if exited_at < self.num_layers:
            for layer_idx in range(exited_at, self.num_layers):
                layer_cache = new_kv_cache.layers[layer_idx]
                layer_cache.keys = layer_cache.keys[:, :, :-1, :]
                layer_cache.values = layer_cache.values[:, :, :-1, :]

        return predicted_id, exited_at, exit_confidence, new_kv_cache

    @torch.no_grad()
    def generate_full_depth(self, input_ids, max_new_tokens=30, eos_token_id=None, verbose=True):
        """
        Native full-depth decoding path.
        Use this when only the final layer is enabled as an exit to guarantee
        parity with standard greedy decoding.
        """

        generated_ids = input_ids.clone()
        exit_history = []
        exit_counts = {self.num_layers: 0}
        total_generated_layer_executions = 0

        outputs = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        for step in range(max_new_tokens):
            probs = F.softmax(logits, dim=-1)
            confidence, next_token = torch.max(probs, dim=-1)
            next_token_id = next_token.item()

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

            token_string = tokenizer.decode([next_token_id])
            exit_history.append(
                {
                    "token_index": step,
                    "token_id": next_token_id,
                    "token": token_string,
                    "exit_layer": self.num_layers,
                    "confidence": confidence.item(),
                }
            )

            exit_counts[self.num_layers] += 1
            total_generated_layer_executions += self.num_layers

            if verbose:
                print(f"  Generated: {repr(token_string)}")
                print(f"  Exit layer: {self.num_layers}")
                print(f"  Confidence: {confidence.item():.4f}")

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            outputs = self.model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        return generated_ids, exit_history, exit_counts, total_generated_layer_executions

    @torch.no_grad()
    def generate_early_exit_pruned(self, input_ids, max_new_tokens=30, eos_token_id=None, verbose=True):
        if EXIT_LAYERS == [self.num_layers]:
            if verbose:
                print("\n", "=" * 80, "FULL-DEPTH FALLBACK (native model cache)", "=" * 80, sep="\n")
            return self.generate_full_depth(input_ids, max_new_tokens, eos_token_id, verbose)

        device = input_ids.device
        if verbose:

            print("\n", "=" * 80, "PROMPT PREFILL", "=" * 80, sep="\n")

        kv_cache, _, _, _, _ = self.prefill_prompt(input_ids, False)
        first_token_id, first_exit_layer, first_confidence = self.select_first_token_from_model_forward(input_ids, verbose)

        generated_ids = input_ids.clone()

        exit_history = []

        exit_counts = {layer: 0 for layer in EXIT_LAYERS}
        if self.num_layers not in exit_counts:
            exit_counts[self.num_layers] = 0


        total_generated_layer_executions = 0

        next_token = torch.tensor([[first_token_id]], device=device, dtype=torch.long)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        position = input_ids.shape[1]

        if verbose:
            print("\nProcessing generated token #1")
            print(f"  Generated: {repr(tokenizer.decode(next_token[0]))}")
            print(f"  Exit layer: {first_exit_layer}")
            print(f"  Confidence: {first_confidence:.4f}")

        predicted_next, exit_layer, confidence, kv_cache = self.generate_one_token(next_token, position, kv_cache, verbose)
        total_generated_layer_executions += first_exit_layer
        exit_counts[first_exit_layer] += 1

        exit_history.append(
            {
                "token_index": 0,
                "token_id": next_token.item(),
                "token": tokenizer.decode(next_token[0] ),
                "exit_layer": first_exit_layer,
                "confidence": first_confidence,
            }    )

        if ( eos_token_id is not None and next_token.item() == eos_token_id):
            return generated_ids, exit_history, exit_counts, total_generated_layer_executions

        for step in range(1, max_new_tokens):
            if verbose:
                print("\n", "-" * 80, f"GENERATED TOKEN #{step + 1}", "-" * 80, sep="\n")

            token_id = torch.tensor([[predicted_next]], device=device, dtype=torch.long)
            current_token_id = token_id.item()

            position = input_ids.shape[1] + step

            predicted_next, exit_layer, confidence, kv_cache = self.generate_one_token(token_id, position, kv_cache, verbose)

            total_generated_layer_executions += exit_layer

            exit_counts[exit_layer] += 1

            generated_ids = torch.cat([generated_ids, token_id], dim=-1)

            token_string = tokenizer.decode([current_token_id])

            exit_history.append(
                {
                    "token_index": step,
                    "token_id": current_token_id,
                    "token": token_string,
                    "exit_layer": exit_layer,
                    "confidence": confidence,
                }
            )

            if verbose:
                print(f"  Generated: {repr(token_string)}")
                print(f"  Exit layer: {exit_layer}")
                print(f"  Confidence: {confidence:.4f}")

            if (eos_token_id is not None and current_token_id == eos_token_id):
                break

        return generated_ids, exit_history, exit_counts, total_generated_layer_executions


def load_model():
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    print(f"Loading fine-tuned model from:\n{FINETUNED_MODEL_PATH}")
    model = SmolLM2EarlyExitForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


class SmolLM2EarlyExitForCausalLM(LlamaForCausalLM):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.copy_exit_norms_from_final_norm()
        return model

    def __init__(self, config):

        super().__init__(config)

        if config.num_hidden_layers != 30:
            raise ValueError("Expected exactly 30 layers." )

        self.exit_norm_10 = type(self.model.norm)(config.hidden_size, eps=config.rms_norm_eps)
        self.exit_norm_20 = type(self.model.norm)(config.hidden_size, eps=config.rms_norm_eps)

        with torch.no_grad():
            self.exit_norm_10.weight.copy_(self.model.norm.weight)
            self.exit_norm_20.weight.copy_(self.model.norm.weight)

    def copy_exit_norms_from_final_norm(self):
        with torch.no_grad():
            self.exit_norm_10.weight.copy_( self.model.norm.weight )
            self.exit_norm_20.weight.copy_( self.model.norm.weight )


if __name__ == "__main__":

    tokenizer, model = load_model()

    prompt = [
        {
        "content": "Hey!",
        "role": "user"
    },
    {
        "content": "Hello! How can I help you today?",
        "role": "assistant"
    },
    {
        "content": "I'm planning a trip to Paris. What are some popular tourist attractions?",
        "role": "user"
    }
        ]
    inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    print("\n", "=" * 80, "INPUT", "=" * 80, sep="\n")
    print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
    print(f"\nPrompt length: {inputs['input_ids'].shape[1]} tokens")

    variable_depth_model = VariableDepthSmolLM2(model)

    output_ids, exit_history, exit_counts, total_layer_executions = variable_depth_model.generate_early_exit_pruned(inputs["input_ids"], MAX_NEW_TOKENS, tokenizer.eos_token_id, True)

    print("\n", "=" * 80, "FINAL OUTPUT", "=" * 80, sep="\n")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print("\n", "=" * 80, "EXIT STATISTICS", "=" * 80, sep="\n")

    for layer in EXIT_LAYERS:

        count = exit_counts[layer]

        print(f"Layer {layer:2d}: {count} tokens")

    generated_count = len(exit_history)

    print(f"\nGenerated tokens: {generated_count}")

    full_layer_executions = generated_count * MAX_NEW_TOKENS

    if full_layer_executions > 0:
        saving = ( 1.0 -(total_layer_executions / full_layer_executions)) * 100.0
    else:
        saving = 0.0

    print(f"Variable-depth layer executions: {total_layer_executions}")
    print(f"Full-depth layer executions: {full_layer_executions}")
    print(f"Estimated layer-compute reduction: {saving:.2f}%")
    print("\n", "=" * 80, "PER-TOKEN EXIT INFORMATION", "=" * 80, sep="\n")

    for item in exit_history:
        print(f"{item['token_index']:3d} | {repr(item['token']):20s} | exit={item['exit_layer']:2d} | confidence={item['confidence']:.4f}")
