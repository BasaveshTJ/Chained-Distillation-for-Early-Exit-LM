# Chained Distillation For Early Exit Transformer

This repository explores adaptive computation for decoder-only language models using two directions:

- Early exit with chained distillation.
- Deep-end selective adaptation (DeepAdapt).

The goal is to reduce compute for easy tokens while preserving generation quality.

## Core Idea

In standard Transformers, every token goes through all layers. This project assumes many tokens are easy and can be predicted reliably from shallower layers.

Chained distillation is used so shallow layers learn from deeper layers:

Loss = KL(L4, L8) + KL(L8, L12) + CE(L12, GroundTruth)

where L4, L8, and L12 represent the outputs of layers 4, 8, and 12 respectively, and GroundTruth represents the true labels for the tokens.
The final layer is supervised using the ground-truth labels. Intermediate layers are trained to match the hidden representation's classifier outputs of deeper layers. The distillation loss is weighted by the prediction confidence of the deeper layer. Higher confidence corresponds to a larger weight for that loss, encouraging shallow layers to accurately predict tokens that are already easy for the deeper model. As a result, high-confidence tokens can be predicted accurately by earlier layers using the same classifier head. 

At inference, confidence at exit layers is used as the routing signal:

- If confidence is above a threshold, the token exits early.
- Otherwise, computation continues to deeper layers.

## What Was Done

- Fine-tuned SmolLM2-135M-Instruct with chained distillation on smol-smoltalk everyday-conversations.
- Evaluated multiple exit sets: [10, 20, 30], [20, 30], and [30].
- Compared early-exit behavior against non-early-exit generation.
- Added and tested KV-cache pruning for early-exited tokens.
- Ran DeepAdapt by freezing the base stack up to layer 30 and training one new deep layer (layer 31) with confidence-gated routing.

## High-Level Outcomes

- Early-exit outputs remained close to non-early-exit outputs on the tested prompt, indicating functional early exits with limited qualitative drop in that setup.
- Token routing showed most tokens can exit early in some runs, while harder tokens continue deeper.
- Naive KV-cache pruning preserved quality only in the first part of generation and then drifted, suggesting deeper layers still rely on historical KV from earlier tokens.
- DeepAdapt DeepAdapt, where based model was trained on instruct data with confidence-gated routing,produced generally more assistant-like responses than the non-instruct base behavior, which showed that selective deep-layer training can effectively adapt the model to new tasks while maintaining their core capabilities.
- Current evidence is qualitative and prompt-limited; broader evaluation is required for strong claims on quality and speedup.
- without pruning the KV-cache, early-exit can't be fully effective as deeper layers still rely on historical KV from earlier tokens, so pre-training with appropriate mask attention can help mitigate this dependency and improve early-exit efficiency.
- deepadapt can still be limited, especially when new data that is learned contains updated knowledge that conflicts with the frozen base layers. in this if the model exit at early layers it's predictions will not reflect the new knowledge. This limitation can be mitigated by ensuring that the shallow layers learn only fundamental grammar and relatively stable knowledge, while the deeper layers remain responsible for adapting to new and evolving information.

## Why This Matters

- Supports adaptive compute: easy tokens can be processed cheaply.
- Preserves a path for harder tokens to use deeper reasoning layers.
- Opens a parameter-efficient adaptation strategy using selective deep-layer training.

## Full Details

experiment, observations, and file references are in [src/outcomes.md](src/outcomes.md).