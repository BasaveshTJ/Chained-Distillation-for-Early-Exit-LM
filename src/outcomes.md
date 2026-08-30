## Experiment Outcome

### Early Exit with chained distillation
Fine-tuning with chained distillation was run with:

- MODEL_ID: HuggingFaceTB/SmolLM2-135M-Instruct
- DATASET_ID: HuggingFaceTB/smol-smoltalk (everyday-conversations)
- EXIT_LAYERS: [10, 20, 30]
- TEMPERATURE: 2.0
- LAMBDA_KD: [1.0, 1.0]
- LAMBDA_CE: 1.0
- learning_rate: 2e-4
- epochs: 10
- loss: KL(L4, L8) + KL(L8, L12) + CE(L12, GroundTruth)

after fine tuning, the model was evaluated on a conversation prompt with max_new_tokens=30 and threshold=0.6, using three exit configurations: [10, 20, 30], [20, 30], and [30]. 
 
and below are the observed outcomes:

For the tested conversation prompt with max_new_tokens=30, early-exit outputs stayed close to non-early-exit outputs across all three exit configurations, suggesting functional early exits with limited qualitative degradation on the tested prompt.

In the [10, 20, 30] setup, most tokens exited early, with fewer tokens requiring deeper layers. One representative run showed layer 10 = 22 tokens, layer 20 = 6 tokens, and layer 30 = 2 tokens.

Layer-wise token comparison showed that early-exit and non-early-exit top-5 candidates were different, especially after layer 10. In several shallow-layer checks, early-exit candidates appeared more aligned with the prompt context, this should be treated as a qualitative observation because alignment was not measured with a formal metric.

It was also observed that earlier checkpoints with early exit were not able to produce meaningful output under shallow exits, while their last layer still gave close-to-expected output. But the later checkpoints improved shallow exits and moved closer to expected full-response behavior.

in above test even with early exit, the model was still executing all layers per token, which can increase compute due to repeated exit-head checks. so pruning was tested, to check if model can preserve the same quality while reducing compute by removing KV caches for tokens that exited early.

in this test, the input prompt was executed for all layers, but for the generated tokens, when a token exited at layer k, its KV was retained up to layer k and removed from deeper-layer caches. 

it was observed that pruning did not significantly affect the quality of the first few generated tokens, but after certain token like around 10, the generated text started to drift away from the contetxt of the prompt, and the quality of the generated text was not preserved. This can be interpreted as progressive degradation from naive KV-cache pruning, suggesting that deeper-layer generation still depends on historical KV information from tokens that exited earlier. Since pruning-aware behavior was not part of training, this outcome was expected.

with these observations, it was seen that the model's shallow layers were able to learn to predict deeper-layer representations, and the training logic where shallow layers learns from deeper layers instead of ground truth supported stable early-exit behavior on the tested prompt.


This initial experiment demonstrated that a chained-distillation loss allows shallow layers to effectively learn and mimic deeper-layer representations, enabling functional early-exits. Since the training was run for 10 epochs on a small dataset, overfitting is a significant concern. To transition this experiment into a full evaluation, this experiment should be scaled up to a complete, diverse dataset like smol-smoltalk, with training restricted to 1–2 epochs, mirroring standard pre-training. This would help determine if the early-exit behavior holds up without memorizing data samples, and also evaluation on other exit criteria instead of hard thresholds also helps to understand the model's behavior better. For pruning, the model should be trained with pruning-aware behavior, so that the model can learn to preserve quality even with pruning. Finally, a definitive evaluation requires tracking hard execution metrics—such as Time to First Token, inter token latency, and FLOP reductions to measure the exact speedup against standard non-early-exit generation.


### DeepAdapt Experiment

The DeepAdapt experiment was designed to test whether task adaptation can be done efficiently by preserving the base model knowledge and only training a small extension at the deep end. The idea was to use SmolLM2-135M (base) as a frozen backbone up to the 30th layer, add one new trainable layer as the 31st layer with its own classifier head, and update this new layer only when the frozen 30th-layer prediction was uncertain (confidence < 0.5). For this setup, fine-tuning was run on the everyday-conversations subset of smol-smoltalk for 3 epochs with learning rate 2e-4, and the training loss was applied only to tokens whose confidence at exit 30 was below 0.5.

In inference with greedy decoding and confidence-gated routing (exit at 30 if confidence >= 0.5, otherwise route to 31), the adapted model generally produced responses that were more elaborative and assistant like than the base model which was only pre-trained and not fine tuned on instruct. Qualitatively, the generation stayed on-topic, and the routing behavior showed that the model still used the frozen exit for some of the tokens while escalating uncertain token predictions to the new deep layer. In one representative run, 30 generated tokens were produced, with 9 tokens exited at exit 30 and 21 routed to exit 31.

These results suggest that the added deep layer can change and specialize the response behavior while the frozen backbone still provides the core language prior, supporting the idea that selective deep end new layer adaptation can be a practical path for low-data and lower-update fine-tuning.

This direction is parameter-efficient, preserves most pretrained knowledge through freezing, and offers an interpretable routing signal through confidence-based escalation. Fixed-threshold confidence gating can still be unstable across prompts, selective-token training can bias learning toward uncertain regions while neglecting global fluency consistency, and the current evaluation is narrow, so broader datasets and early exit pre-trained models need to be evaluated to assess the quality and efficiency of this approach.


- Early-exit baseline without pruning: [src/inference/inference_ee.py](src/inference/inference_ee.py)
- Early-exit with KV-cache pruning logic: [src/inference/inference_ee_prune.py](src/inference/inference_ee_prune.py)
- Standard full-generation reference: [src/inference/inference.py](src/inference/inference.py)
- Layer-wise token inspection: [src/inference/inference_layers.py](src/inference/inference_layers.py)
-  deep-adapt : [src/inference/inference_deepadapt.py](src/inference/inference_deepadapt.py)