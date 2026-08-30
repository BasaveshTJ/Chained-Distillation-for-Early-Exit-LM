from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.utils.generic import ModelOutput


@dataclass
class DeepAdaptCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    base_logits: Optional[torch.FloatTensor] = None
    low_conf_mask: Optional[torch.Tensor] = None
    num_valid_tokens: Optional[torch.Tensor] = None
    entropy_sum: Optional[torch.Tensor] = None
    num_correct_tokens: Optional[torch.Tensor] = None
    past_key_values: Optional[object] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class DeepAdaptResidualLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.input_norm = nn.RMSNorm(hidden_size, eps=eps)
        self.fc_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.fc_up(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.fc_down(hidden_states)
        return residual + hidden_states


class SmolLM2DeepAdaptForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.confidence_threshold = 0.5

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        self.deepadapt_layer = DeepAdaptResidualLayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            eps=config.rms_norm_eps,
        )
        self.deepadapt_norm = type(self.model.norm)(config.hidden_size, eps=config.rms_norm_eps)
        self.deepadapt_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        with torch.no_grad():
            self.deepadapt_norm.weight.copy_(self.model.norm.weight)
            self.deepadapt_lm_head.weight.copy_(self.lm_head.weight)

    def set_confidence_threshold(self, threshold: float):
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("confidence threshold must be in (0, 1).")
        self.confidence_threshold = threshold

    def verify_trainable_parameters(self):
        print("=" * 70)
        print("DeepAdapt trainable parameter verification")
        print("=" * 70)

        trainable_names = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad
        ]

        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)

        print("Trainable modules:")
        for name in trainable_names:
            print(f"- {name}")

        ratio = 100.0 * trainable_params / max(total_params, 1)
        print(f"\nTrainable params: {trainable_params:,}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable ratio: {ratio:.4f}%")
        print("=" * 70)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=False,
        return_dict=True,
        cache_position=None,
        **kwargs,
    ):
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = base_outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("hidden_states is None while output_hidden_states=True.")

        frozen_hidden_30 = hidden_states[30].detach()

        with torch.no_grad():
            base_logits = self.lm_head(frozen_hidden_30)
            top_logits = base_logits.max(dim=-1).values
            lse = torch.logsumexp(base_logits, dim=-1)
            base_confidence = torch.exp(top_logits - lse)
            low_conf_mask = base_confidence < self.confidence_threshold

        adapted_hidden = self.deepadapt_layer(frozen_hidden_30)
        adapted_hidden = self.deepadapt_norm(adapted_hidden)
        adapted_logits = self.deepadapt_lm_head(adapted_hidden)

        num_valid_tokens = torch.zeros((), device=adapted_logits.device, dtype=torch.long)
        entropy_sum = torch.zeros((), device=adapted_logits.device, dtype=adapted_logits.dtype)
        num_correct_tokens = torch.zeros((), device=adapted_logits.device, dtype=torch.long)

        if labels is not None:
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels != -100
            shift_low_conf_mask = low_conf_mask[:, :-1] & valid_mask
            num_valid_tokens = shift_low_conf_mask.sum()

            shift_logits = adapted_logits[..., :-1, :]
            valid_shift_labels = shift_labels.clamp_min(0)
            token_log_probs = F.log_softmax(shift_logits, dim=-1)
            token_probs = token_log_probs.exp()
            token_entropy = -(token_probs * token_log_probs).sum(dim=-1)
            entropy_sum = (token_entropy * shift_low_conf_mask.to(dtype=token_entropy.dtype)).sum()
            num_correct_tokens = ((shift_logits.argmax(dim=-1) == valid_shift_labels) & shift_low_conf_mask).sum()
        elif attention_mask is not None:
            num_valid_tokens = attention_mask[:, 1:].to(dtype=torch.long).sum()

        return DeepAdaptCausalLMOutput(
            loss=None,
            logits=adapted_logits,
            base_logits=base_logits,
            low_conf_mask=low_conf_mask,
            num_valid_tokens=num_valid_tokens,
            entropy_sum=entropy_sum,
            num_correct_tokens=num_correct_tokens,
            past_key_values=base_outputs.past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=base_outputs.attentions,
        )
