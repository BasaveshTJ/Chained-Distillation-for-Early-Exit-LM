import torch
import torch.nn.functional as F


def hierarchical_early_exit_kd_loss(
    exit_logits,
    labels,
    temperature=2.0,
    lambda_kd=None,
    lambda_ce=1.0,
    vocab_chunk_size=1024,
):
    """
    Hierarchical confidence-weighted knowledge distillation.

    Architecture:

        EXIT 10 -> EXIT 20 -> EXIT 30

    Loss:

        lambda_kd[0] * KL(EXIT10 || EXIT20)
      + lambda_kd[1] * KL(EXIT20 || EXIT30)
      + lambda_ce    * CE(EXIT30, ground_truth)

    The teacher is detached.

    IMPORTANT:
    labels provided by SFTTrainer are NOT already shifted.

    Therefore:

        logits[..., :-1, :]
        labels[..., 1:]

    are used here for causal next-token prediction.

    The KL computation is performed in vocabulary chunks to avoid
    materializing large [B, S, V] softmax/log-softmax tensors.
    """

    num_exits = len(exit_logits)

    if num_exits < 2:
        raise ValueError("Need at least two exits for distillation.")

    if lambda_kd is None:
        lambda_kd = [1.0] * (num_exits - 1)

    if len(lambda_kd) != num_exits - 1:
        raise ValueError(
            "lambda_kd must contain exactly "
            f"{num_exits - 1} values."
        )

    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    # ---------------------------------------------------------
    # Causal shift
    #
    # Input:
    #
    # logits:
    #   position 0 predicts label 1
    #   position 1 predicts label 2
    #   ...
    #
    # Therefore:
    #
    # logits[..., :-1, :]
    # labels[..., 1:]
    # ---------------------------------------------------------

    shift_labels = labels[..., 1:]

    valid_mask = shift_labels != -100

    # [B, S-1]
    valid_mask_float = valid_mask.to(dtype=exit_logits[0].dtype)

    valid_count = valid_mask_float.sum()

    # Avoid division by zero.
    valid_count = valid_count.clamp_min(1.0)

    total_loss = exit_logits[0].new_zeros(())

    # ---------------------------------------------------------
    # Hierarchical KD
    # ---------------------------------------------------------

    for i in range(num_exits - 1):
        student_logits = exit_logits[i][..., :-1, :]
        teacher_logits = exit_logits[i + 1][..., :-1, :]
        vocab_size = student_logits.size(-1)

        # -----------------------------------------------------
        # Teacher is always detached.
        #
        # EXIT 20 teaches EXIT 10.
        #
        # EXIT 30 teaches EXIT 20.
        #
        # The deeper exit does NOT receive gradient from KD.
        # -----------------------------------------------------

        teacher_logits = teacher_logits.detach()

        # -----------------------------------------------------
        # Compute logsumexp over the entire vocabulary.
        #
        # This creates only [B,S], NOT [B,S,V].
        # -----------------------------------------------------

        with torch.no_grad():
            teacher_lse = torch.logsumexp(teacher_logits / temperature, dim=-1)

        student_lse = torch.logsumexp(student_logits / temperature, dim=-1)

        # -----------------------------------------------------
        # Teacher confidence for the correct token.
        #
        # p_teacher(y)
        #
        # Only the probability of the ground-truth is needed
        # token, so there is no need to construct the complete
        # probability distribution.
        # -----------------------------------------------------

        teacher_target_logits = teacher_logits.gather(dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            teacher_target_log_prob = (teacher_target_logits / temperature) - teacher_lse

            confidence = teacher_target_log_prob.exp()

            confidence = confidence * valid_mask_float

        # -----------------------------------------------------
        # KL divergence.
        #
        # KL(P_teacher || P_student)
        #
        # = sum_v P_teacher(v)
        #     [log P_teacher(v) - log P_student(v)]
        #
        # Compute it in vocabulary chunks.
        #
        # This avoids:
        #
        #   softmax -> [B,S,V]
        #   log_softmax -> [B,S,V]
        #   kl_div -> [B,S,V]
        #
        # simultaneously occupying GPU memory.
        # -----------------------------------------------------

        kd_loss = student_logits.new_zeros(student_logits.shape[:-1])

        for start in range(0, vocab_size, vocab_chunk_size):
            end = min(start + vocab_chunk_size, vocab_size)

            student_chunk = student_logits[..., start:end] / temperature

            teacher_chunk = teacher_logits[..., start:end] / temperature

            student_log_probs = student_chunk - student_lse.unsqueeze(-1)

            with torch.no_grad():
                teacher_probs = torch.exp(teacher_chunk - teacher_lse.unsqueeze(-1))
                teacher_log_probs = teacher_chunk - teacher_lse.unsqueeze(-1)

            kd_loss += (
                teacher_probs
                * (
                    teacher_log_probs
                    - student_log_probs
                )
            ).sum(dim=-1)

        # -----------------------------------------------------
        # Confidence weighting.
        # -----------------------------------------------------

        weighted_kd = kd_loss * confidence

        # Mask padding / ignored positions.
        weighted_kd = weighted_kd * valid_mask_float

        pair_loss = weighted_kd.sum() / valid_count

        total_loss += lambda_kd[i] * (temperature ** 2) * pair_loss

    # ---------------------------------------------------------
    # Final supervised CE
    # ---------------------------------------------------------

    final_logits = exit_logits[-1][..., :-1, :]

    ce_loss = F.cross_entropy(
        final_logits.reshape(
            -1,
            final_logits.size(-1),
        ),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

    ce_loss = ce_loss / valid_count

    total_loss += lambda_ce * ce_loss

    return total_loss