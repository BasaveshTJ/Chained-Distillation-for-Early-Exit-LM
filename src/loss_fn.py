import torch
import torch.nn.functional as F


def hierarchical_early_exit_kd_loss(exit_logits, labels, temperature=2.0, lambda_kd=None, lambda_ce=1.0):
    """
    Hierarchical knowledge distillation loss for early exits.

    Args:
        exit_logits: List of logits from exit layers. Each element: Tensor [batch, seq_len, vocab_size], ordered shallow -> deep.
        labels: Ground truth next-token labels. Tensor [batch, seq_len]
        temperature: Temperature for soft targets.
        lambda_kd: List of KD weights, length num_exits-1. Example: [1.0, 1.0, 1.0]
        lambda_ce: Weight for final exit supervised CE loss.

    Returns:
        Scalar loss
    """
    num_exits = len(exit_logits)

    assert num_exits >= 2, "Need at least two exits for distillation"

    if lambda_kd is None:
        lambda_kd = [1.0] * (num_exits - 1)

    assert len(lambda_kd) == num_exits - 1, "lambda_kd length must equal number of KD pairs"

    total_loss = 0.0

    # Hierarchical KD losses: shallow exit learns from next deeper exit
    for i in range(num_exits - 1):
        student_logits = exit_logits[i]
        teacher_logits = exit_logits[i + 1]

        # soft distributions
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        # KL(student || teacher), sum over vocabulary -> shape: [batch, seq_len]
        kd_loss = F.kl_div(student_log_probs, teacher_probs.detach(), reduction="none").sum(dim=-1)

        # teacher confidence on correct token
        confidence = teacher_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # confidence weighted KD
        weighted_kd = kd_loss * confidence

        total_loss += lambda_kd[i] * (temperature ** 2) * weighted_kd.mean()

    # Final exit supervised loss
    final_logits = exit_logits[-1]
    ce_loss = F.cross_entropy(
        final_logits.reshape(-1, final_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100
    )

    total_loss += lambda_ce * ce_loss

    return total_loss


# outputs = model(input_ids, output_exit_logits=True)
# exit_logits = outputs.exit_logits
# exit_logits = [
#     logits_layer4,    # [B,S,V]
#     logits_layer8,    # [B,S,V]
#     logits_layer12    # [B,S,V]
# ]
# loss = hierarchical_early_exit_kd_loss(
#     exit_logits=exit_logits, labels=labels, temperature=2.0, lambda_kd=[1.0, 1.0], lambda_ce=1.0
# )
# loss.backward()
