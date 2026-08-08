# Chained-distillation-for-early-exit-transformer

Large language models built on the Transformer architecture process every token through all layers. However, many tokens are straightforward to predict and may not require the full depth of the network. This is evident when comparing small and large language models: smaller models often generate reasonable responses for simple tasks but struggle with complex reasoning, while larger models perform better on difficult tasks.

Most generated tokens, however, are relatively simple, such as common words, grammatical fillers, and punctuation. These tokens may not require deep computation. Early exiting is therefore a promising approach to reduce computation by predicting such tokens using shallower layers. The key challenge is determining which tokens are easy enough to exit early.

This decision can be learned by the model itself. The idea is to start with a pretrained model and train the shallow layers to predict the representations of deeper layers. A hierarchical knowledge distillation objective can be used, for example:

Loss = KL(L4, L8) + KL(L8, L12) + CE(L12, GroundTruth)

where:

The final layer is supervised using the ground-truth labels.
Intermediate layers are trained to match the hidden representation's classifier outputs of deeper layers.
The distillation loss is weighted by the prediction confidence of the deeper layer. Higher confidence corresponds to a larger weight, encouraging shallow layers to accurately predict tokens that are already easy for the deeper model.
As a result, high-confidence tokens can be predicted accurately by earlier layers using the same classifier head. During inference, this shared classifier head can also act as an exit gate: if the prediction confidence exceeds a threshold, the token is emitted immediately; otherwise, computation continues to the next exit layer.
## Benefits
- Reduced computation by avoiding unnecessary deep-layer processing for easy tokens.
- Faster inference with minimal impact on prediction quality.
- Adaptive computation, where difficult tokens receive more processing while easy tokens exit early.





Can early exit make small models smarter by using the parameters efficiently, by using shallow layers for simple tasks and deeper layers for more reasoning, and also pruning?

Will this make fine-tuning a model for a specific task easier, where a model that has been trained for early exit is used, with a few extra trainable layers added at the deeper end and only those layers are trained, while the basic knowledge is still preserved by the frozen layers, and the model is able to learn the new task with less data and less training time?