from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.utils.generic import ModelOutput


@dataclass
class EarlyExitCausalLMOutput(ModelOutput):
    """
    Output for SmolLM2 with early exits.

    exit_logits:
        [
            logits from layer 10,
            logits from layer 20,
            logits from layer 30,
        ]

    All three logits use the SAME lm_head.

    exit 10 -> exit_norm_10 -> shared lm_head
    exit 20 -> exit_norm_20 -> shared lm_head
    exit 30 -> model.norm   -> shared lm_head
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    exit_logits: Optional[List[torch.FloatTensor]] = None
    past_key_values: Optional[object] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    num_valid_tokens: Optional[torch.Tensor] = None
    entropy_sum: Optional[torch.Tensor] = None
    num_correct_tokens: Optional[torch.Tensor] = None


class SmolLM2EarlyExitForCausalLM(LlamaForCausalLM):
    EXIT_LAYERS = [10, 20, 30]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.copy_exit_norms_from_final_norm()
        return model

    def __init__(self, config):
        super().__init__(config)

        if config.num_hidden_layers != 30:
            raise ValueError(
                "SmolLM2EarlyExitForCausalLM expects exactly 30 "
                f"transformer layers, got {config.num_hidden_layers}."
            )

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # These are SEPARATE normalization modules.
        #
        # They are NOT shared.
        #
        # However, both are initialized from the pretrained
        # final model.norm weights.
        # ---------------------------------------------------------

        self.exit_norm_10 = type(self.model.norm)(config.hidden_size, eps=config.rms_norm_eps)
        self.exit_norm_20 = type(self.model.norm)(config.hidden_size, eps=config.rms_norm_eps)

        # ---------------------------------------------------------
        # Copy the pretrained final RMSNorm weights.
        #
        # This is a COPY, not weight sharing.
        # ---------------------------------------------------------

        with torch.no_grad():
            self.exit_norm_10.weight.copy_(self.model.norm.weight)
            self.exit_norm_20.weight.copy_(self.model.norm.weight)

        # ---------------------------------------------------------
        # The classifier is NOT duplicated.
        #
        # self.lm_head is inherited from LlamaForCausalLM.
        #
        # It remains the SAME Linear module for all exits.
        # ---------------------------------------------------------

        if self.lm_head.weight.shape != (
            config.vocab_size,
            config.hidden_size,
        ):
            raise ValueError(
                "Unexpected lm_head shape: "
                f"{tuple(self.lm_head.weight.shape)}"
            )

    def copy_exit_norms_from_final_norm(self):
        with torch.no_grad():
            self.exit_norm_10.weight.copy_(self.model.norm.weight)
            self.exit_norm_20.weight.copy_(self.model.norm.weight)

    def verify_architecture(self):
        print("=" * 70)
        print("Early-exit architecture verification")
        print("=" * 70)

        print(f"Transformer layers: {self.config.num_hidden_layers}")
        print(f"Exit layers: {self.EXIT_LAYERS}")

        print("\n## CLASSIFIER")

        print(f"Shared LM head: {type(self.lm_head).__name__}")
        print(f"LM head shape: {tuple(self.lm_head.weight.shape)}")

        print("Input embedding == LM head:", self.lm_head.weight.untyped_storage().data_ptr() == self.model.embed_tokens.weight.untyped_storage().data_ptr())

        print("\n## NORMALIZATION")
        print("Exit 10 norm:", type(self.exit_norm_10).__name__)
        print("Exit 20 norm:", type(self.exit_norm_20).__name__)
        print("Exit 30 norm:", type(self.model.norm).__name__)

        # These MUST be different Parameter objects.
        print("Exit 10 norm == Exit 20 norm:", self.exit_norm_10.weight.data_ptr() == self.exit_norm_20.weight.data_ptr())
        print("Exit 10 norm == Exit 30 norm:", self.exit_norm_10.weight.data_ptr() == self.model.norm.weight.data_ptr())
        print("Exit 20 norm == Exit 30 norm:", self.exit_norm_20.weight.data_ptr() == self.model.norm.weight.data_ptr())

        # But initially their values should be identical.
        print("Exit 10 norm initially copied from pretrained norm:", torch.equal(self.exit_norm_10.weight.detach(), self.model.norm.weight.detach()))
        print("Exit 20 norm initially copied from pretrained norm:", torch.equal(self.exit_norm_20.weight.detach(), self.model.norm.weight.detach()))

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
        # ---------------------------------------------------------
        # We need hidden states at layers 10 and 20.
        #
        # We do NOT expose all hidden states by default because
        # that wastes memory.
        # ---------------------------------------------------------

        outputs = super().forward(
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

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("hidden_states is None while output_hidden_states=True.")

        # ---------------------------------------------------------
        # EXIT 10
        #
        # hidden_states[10] =
        # output after transformer layer 10,
        # before final pretrained model.norm.
        #
        # Apply a separate exit_norm_10.
        # ---------------------------------------------------------

        hidden_10 = hidden_states[10]
        normalized_10 = self.exit_norm_10(hidden_10)
        logits_10 = self.lm_head(normalized_10)

        # ---------------------------------------------------------
        # EXIT 20
        # ---------------------------------------------------------

        hidden_20 = hidden_states[20]
        normalized_20 = self.exit_norm_20(hidden_20)
        logits_20 = self.lm_head(normalized_20)

        # ---------------------------------------------------------
        # EXIT 30
        #
        # hidden_states[30] is already normalized by model.norm.
        # ---------------------------------------------------------

        hidden_30 = hidden_states[30]
        logits_30 = self.lm_head(hidden_30)

        # ---------------------------------------------------------
        # Shared classifier
        # ---------------------------------------------------------

        exit_logits = [logits_10, logits_20, logits_30]

        # ---------------------------------------------------------
        # Number of valid labels.
        #
        # TRL's current SFTTrainer expects this field when using
        # its token-count normalization path.
        #
        # The actual causal shift is handled by loss_fn.py.
        # ---------------------------------------------------------

        num_valid_tokens = torch.zeros((), device=logits_30.device, dtype=torch.long)
        entropy_sum = torch.zeros((), device=logits_30.device, dtype=logits_30.dtype)
        num_correct_tokens = torch.zeros((), device=logits_30.device, dtype=torch.long)
        if labels is not None:
            num_valid_tokens = (labels[:, 1:] != -100).sum()
            shift_logits = logits_30[..., :-1, :]
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels != -100
            valid_shift_labels = shift_labels.clamp_min(0)
            token_log_probs = F.log_softmax(shift_logits, dim=-1)
            token_probs = token_log_probs.exp()
            token_entropy = -(token_probs * token_log_probs).sum(dim=-1)
            entropy_sum = (token_entropy * valid_mask.to(dtype=token_entropy.dtype)).sum()
            num_correct_tokens = ((shift_logits.argmax(dim=-1) == valid_shift_labels) & valid_mask).sum()
        elif attention_mask is not None:
            num_valid_tokens = attention_mask[:, 1:].to(dtype=torch.long).sum()

        # ---------------------------------------------------------
        # Standard final logits
        # ---------------------------------------------------------

        final_logits = logits_30

        return EarlyExitCausalLMOutput(
            loss=None,
            logits=final_logits,
            exit_logits=exit_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            num_valid_tokens=num_valid_tokens,
            entropy_sum=entropy_sum,
            num_correct_tokens=num_correct_tokens,
        )