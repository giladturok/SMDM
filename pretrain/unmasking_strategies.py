"""Deterministic unmasking strategies for masked diffusion models allow for *exact* evaluation of negative log-likelihood of a data sample x. 

At each denoising step, a strategy defines 
(1) which *positions* to unmask
(2) how to determine the *value* the unmasked position takes from their vocabulary distribution
"""
import math

import torch
import torch.nn as nn


def topk_confidence_nll(
    x: torch.Tensor, model: nn.Module, k: int = 1
) -> torch.Tensor:
    """Compute negative log-likelihood of data x under top-k confidence unmasking strategy.

    Assumes that each denoising step unmasks exactly k positions and thus each batch
    element has the same number of masked positions.

    Returns:
        nll [batch_size]: negative log-likelihood averaged over sequence for each batch element
    """
    x = x[:, 0 : model.config.block_size].contiguous() #[batch_size, seq_len]
    batch_size, seq_len = x.shape
    mask_token_id = model.config.vocab_size
    
    nll_topk_per_step, nll_elbo_per_step = [], []
    
    assert k > 0, "k must be positive integer"
    assert k <= seq_len, "k must be less than or equal to sequence length"
    
    z_t = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    log_probs = torch.zeros(batch_size, device=x.device) # [batch_size]

    steps = math.ceil(seq_len / k)
    for t in reversed(range(1, steps + 1)):
        # PREDICT: Get model probabilities
        logits = model(z_t) # [batch_size, seq_len, vocab_size]
        token_log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]

        # SELECT: Identify top-k most confident positions to unmask
        mask = (z_t == mask_token_id)  # [batch_size, seq_len]
        num_masked = mask.sum(dim=-1)  # assumes all in batch have same num masked [1]
        confidence = torch.max(token_log_probs, dim=-1).values  # [batch_size, seq_len]
        confidence_masked = torch.where(mask, confidence, torch.full_like(confidence, float('-inf')))  # [batch_size, seq_len]
        _, selected_positions = torch.topk(
            confidence_masked, k=min(k, num_masked), dim=-1 # adjust k for last step
        ) # [batch_size, k]
        
        # UPDATE: Unmask top-k positions and accumulate log-probabilities
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(-1) # [batch_size, 1]
        true_tokens = x[batch_idx, selected_positions] # [batch_size, k]
        log_probs_at_selected = token_log_probs[batch_idx, selected_positions, true_tokens] # [batch_size, k]
        log_probs += log_probs_at_selected.sum(dim=-1) # [batch_size]
        z_t[batch_idx, selected_positions] = true_tokens # [batch_size, seq_len]
        
        ### DEBUGGING ###
        nll_topk_step = -log_probs_at_selected.mean(dim=-1) # [batch_size]
        nll_topk_per_step.append(nll_topk_step)
        
        log_probs_true_tokens = token_log_probs.gather(
            dim=-1, index=x.unsqueeze(-1)
        ).squeeze(-1) # [batch_size, seq_len]
        # log_probs_masked = torch.where(
        #     mask, log_probs_true_tokens, torch.zeros_like(log_probs_true_tokens)
        # ) # [batch_size, seq_len]
        # mask_prob_fn = lambda t, eps=1e-7: (1 - eps) * t + eps
        # # importance_weight = mask_prob_fn(t / steps) * seq_len 
        # importance_weight = num_masked
        # nll_elbo_step = -log_probs_masked.sum(dim=-1) / importance_weight # [batch_size]
        # nll_elbo_per_step.append(nll_elbo_step)
        
        num_samples=128
        nll_elbo_samples = []
        for _ in range(num_samples):
            # Create random mask with same number of masked tokens
            random_mask = torch.zeros_like(mask) # [batch_size, seq_len]
            for b in range(batch_size):
                # Randomly select num_masked[b] positions to mask
                masked_indices = torch.randperm(seq_len, device=x.device)[:num_masked[b]]
                random_mask[b, masked_indices] = True
            
            # Compute NLL over random mask
            log_probs_random_masked = torch.where(
                random_mask, log_probs_true_tokens, torch.zeros_like(log_probs_true_tokens)
            ) # [batch_size, seq_len]
            nll_elbo_random = -log_probs_random_masked.sum(dim=-1) / num_masked # [batch_size]
            nll_elbo_samples.append(nll_elbo_random) # [batch_size, num_samples]
        nll_elbo_step = torch.stack(nll_elbo_samples).mean(dim=0) # [batch_size]
        nll_elbo_per_step.append(nll_elbo_step)
        
        print(f"Iter {t}\tNLL ELBO: {nll_elbo_step}\tNLL Top-K: {nll_topk_step}")
        
        
    nll = -log_probs / seq_len  # avg NLL per token
    # save conf_selected_list and conf_avg_list to a file for debugging
    # make sure it is saved in a format that is easy to read/parse
    with open("debugging_info.txt", "w") as f:
        f.write("Top-k confidence per step:\n")
        for conf in nll_topk_per_step:
            f.write(f"{conf.item()},\n")
        f.write("Average confidence per step:\n")
        for conf in nll_elbo_per_step:
            f.write(f"{conf.item()},\n")
        
    return nll


def confidence_threshold_nll(
    x: torch.Tensor, 
    model: nn.Module, 
    threshold: float = 0.9, 
    max_steps: int = 2048
) -> float:
    """Compute negative log-likelihood of data x under confidence-threshold unmasking strategy.
    
    At each denoising step, unmask all positions where the model's confidence exceeds the threshold.
    
    Args:
        x: Ground truth tokens [batch_size, seq_len]
        model: The diffusion language model
        threshold: Confidence threshold for unmasking (0 < threshold < 1)
        max_steps: Maximum number of denoising steps (safety limit)
    
    Returns:
        nll: [batch_size] negative log-likelihood averaged over sequence for each batch element
    """
    assert 0.0 < threshold < 1.0, "threshold must be in (0, 1)"
    
    x = x[:, :model.config.block_size].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = x.shape
    mask_token_id = model.config.vocab_size
    log_threshold = math.log(threshold)
    
    # Initialize: all tokens masked
    z_t = torch.full_like(x, mask_token_id)
    log_probs = torch.zeros(batch_size, device=x.device)
    
    # Iterative unmasking
    for step in range(max_steps):
        # Check termination: all tokens unmasked
        is_masked = (z_t == mask_token_id)  # [batch_size, seq_len]
        if not is_masked.any():
            break
        
        # PREDICT: Get model probabilities
        logits = model(z_t)  # [batch_size, seq_len, vocab_size]
        token_log_probs = torch.log_softmax(logits, dim=-1)
        
        # SELECT: Identify positions to unmask based on confidence threshold
        confidence, _ = token_log_probs.max(dim=-1)  # [batch_size, seq_len]
        confidence = confidence.masked_fill(~is_masked, float('-inf'))
        
        # ALWAYS unmask max confidence + anything above threshold (if masked tokens remain)
        above_threshold = (confidence >= log_threshold) # [batch_size, seq_len]
        
        # Max confidence (break ties by selecting first occurrence)
        is_max = (confidence == confidence.amax(dim=1, keepdim=True))
        max_confidence_per_batch = is_max & (is_max.cumsum(dim=1) == 1)
        
        # assert that max confidence per batch has only one True per batch
        assert (max_confidence_per_batch.sum(dim=1) == 1).all()

        # Only unmask if `is_masked` is True (at least one masked token remains)
        to_unmask = (above_threshold | max_confidence_per_batch) & is_masked # [batch_size, seq_len]
        
        # UPDATE: Unmask selected positions and accumulate log-probabilities
        selected_batches, selected_positions = torch.where(to_unmask)
        
        # Gather true tokens and their log probabilities (vectorized)
        true_tokens = x[selected_batches, selected_positions]
        selected_log_probs = token_log_probs[selected_batches, selected_positions, true_tokens]
        
        # Accumulate log probabilities per batch (vectorized)
        log_probs.index_add_(0, selected_batches, selected_log_probs)
        
        # Update masked state
        z_t_minus_1 = z_t.clone()
        z_t_minus_1[selected_batches, selected_positions] = true_tokens
        z_t = z_t_minus_1
    
    # Compute negative log-likelihood
    nll = -log_probs / seq_len  # Average NLL per token
    return nll


def topk_margin_nll(
    x: torch.Tensor, model: nn.Module, k: int = 1
) -> torch.Tensor:
    """Compute negative log-likelihood of data x under top-k margin unmasking strategy.
    
    The margin is defined as the difference between the log-probabilities of the most
    and second-most confident tokens at each position.

    Assumes that each denoising step unmasks exactly k positions and thus each batch
    element has the same number of masked positions.

    Returns:
        nll: [batch_size] negative log-likelihood averaged over sequence for each batch element
    """
    x = x[:, 0 : model.config.block_size].contiguous() #[batch_size, seq_len]
    batch_size, seq_len = x.shape
    mask_token_id = model.config.vocab_size
    
    assert k > 0, "k must be positive integer"
    assert k <= seq_len, "k must be less than or equal to sequence length"
    
    z_t = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    log_probs = torch.zeros(batch_size, device=x.device) # [batch_size]

    steps = math.ceil(seq_len / k)
    for _ in range(steps):
        # PREDICT: Get model probabilities
        logits = model(z_t) # [batch_size, seq_len, vocab_size]
        token_probs = torch.softmax(logits, dim=-1)  # for top-margin computation[batch_size, seq_len, vocab_size]
        token_log_probs = torch.log_softmax(logits, dim=-1)  # for nll computation [batch_size, seq_len, vocab_size]
        
        # SELECT: Identify top-k most confident positions to unmask
        top_two_confidences, _ = torch.topk(token_probs, k=2, dim=-1)  # [batch_size, seq_len, 2]
        most_confident, second_most_confident = top_two_confidences[:, :, 0], top_two_confidences[:, :, 1]  # [batch_size, seq_len]
        margin = most_confident - second_most_confident  # [batch_size, seq_len]
        margin[z_t != mask_token_id] = float('-inf')  # Exclude already unmasked positions
        num_masked = (z_t[0] == mask_token_id).sum(dim=-1).item()  # assumes all in batch have same num masked
        _, selected_positions = torch.topk(margin, k=min(k, num_masked), dim=-1) # adjust k for last step # [batch_size, k]
        
        # UPDATE: Unmask top-k positions and accumulate log-probabilities
        # TODO: Vectorize this loop
        z_t_minus_1 = z_t.clone() # [batch_size, seq_len]
        step_log_probs = torch.zeros(batch_size, device=x.device) # [batch_size]
        
        for batch_idx in range(batch_size):
            for k_idx in range(min(k, num_masked)):
                pos = selected_positions[batch_idx, k_idx].item()
                true_token = x[batch_idx, pos].item()
                step_log_probs[batch_idx] += token_log_probs[batch_idx, pos, true_token]
                z_t_minus_1[batch_idx, pos] = true_token
        
        log_probs += step_log_probs
        z_t = z_t_minus_1
        
    nll = -log_probs / seq_len  # avg NLL per token
    return nll


def block_topk_confidence_nll(
    x: torch.Tensor, model: nn.Module, k: int = 1, block_size: int = 32
) -> torch.Tensor:
    """Compute negative log-likelihood of data x under top-k confidence unmasking strategy.

    At each denoising step, unmask top-k most confident positions *within a block*. By
    restricting unmasking to a specific block, we prevent unmasking from far away 
    positions that are highly confident but likely incorrect.
    
    This topk-block unmasking strategy follows the generative process of Large Language
    Diffusion Models by Nie et al (2025).
    
    Assumes that each denoising step unmasks exactly k positions and thus each batch
    element has the same number of masked positions.

    Returns:
        nll: [batch_size] negative log-likelihood averaged over sequence for each batch element
    """
    x = x[:, 0 : model.config.block_size].contiguous() #[batch_size, seq_len]
    batch_size, seq_len = x.shape
    mask_token_id = model.config.vocab_size
    
    nll_topk_per_step, nll_elbo_per_step = [], []
    
    assert k > 0, "k must be positive integer"
    assert k <= seq_len, "k must be less than or equal to sequence length"
    
    z_t = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    log_probs = torch.zeros(batch_size, device=x.device) # [batch_size]

    num_blocks = math.ceil(seq_len / block_size)
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, seq_len)
        
        steps_in_block = math.ceil((block_end - block_start) / k)
        for t in range(steps_in_block):
            # PREDICT: Get model probabilities
            logits = model(z_t) # [batch_size, seq_len, vocab_size]
            token_log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
            
            # SELECT: Identify top-k most confident postitions to unmask within the block
            masked_tokens = (z_t == mask_token_id) # [batch_size, seq_len]
            seq_idx = torch.arange(seq_len, device=x.device)
            within_block = (seq_idx >= block_start) & (seq_idx < block_end)
            mask = masked_tokens & within_block
            confidence = torch.max(token_log_probs, dim=-1).values # [batch_size, seq_len]
            confidence_masked = torch.where(mask, confidence, torch.full_like(confidence, float("-inf"))) # [batch_size, seq_len]
            num_masked_tokens = masked_tokens.sum(dim=-1)
            _, selected_positions = torch.topk(
                confidence_masked, k=min(k, num_masked_tokens), dim=-1 # adjust k for last step within block
            ) # [batch_size, k]
            
            # UPDATE: Unmask top-k positions and accumulate log-probabilities
            batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(-1) # [batch_size, 1]
            true_tokens = x[batch_idx, selected_positions] # [batch_size, k]
            log_probs_at_selected = token_log_probs[batch_idx, selected_positions, true_tokens] # [batch_size, k]
            log_probs += log_probs_at_selected.sum(dim=-1) # [batch_size]
            z_t[batch_idx, selected_positions] = true_tokens
            
            ### DEBUGGING INFO ###
            nll_topk_step = -log_probs_at_selected.mean(-1) # [batch_size]
            nll_topk_per_step.append(nll_topk_step)
            
            log_probs_true_tokens = token_log_probs.gather(
                dim=-1, index=x.unsqueeze(-1)
            ).squeeze(-1) # [batch_size, seq_len]
            log_probs_masked = torch.where(
                mask, log_probs_true_tokens, torch.zeros_like(log_probs_true_tokens)
            ) # [batch_size, seq_len]
            # num_masked = mask.sum(dim=-1) # [batch_size]
            mask_prob_fn = lambda t, eps=1e-3: (1 - eps) * t + eps
            time_step = t + block_idx * block_size
            total_steps = num_blocks * steps_in_block
            importance_weight = mask_prob_fn(time_step / total_steps)
            nll_elbo_step = -log_probs_masked.sum(dim=-1) / importance_weight # [batch_size]
            nll_elbo_per_step.append(nll_elbo_step)
            
            # print(f"Iter {time_step}\tNLL ELBO: {nll_elbo_step}\tNLL Top-K: {nll_topk_step}")
            
    nll = -log_probs / seq_len  # avg NLL per token
    # make sure it is saved in a format that is easy to read/parse
    # with open("debugging_info.txt", "w") as f:
    #     f.write("Exact NLL per step (with block-greedy decoding):\n")
    #     for conf in nll_topk_per_step:
    #         f.write(f"{conf.item()},\n")
    #     f.write("Average NLL per step (with time-weighting):\n")
    #     for conf in nll_elbo_per_step:
    #         f.write(f"{conf.item()},\n")
    return nll


def topk_autoregressive_nll(
    x: torch.Tensor, model: nn.Module, k: int = 1
) -> torch.Tensor:
    """Compute negative log-likelihood of data x under autoregressive unmasking strategy.

    At each denoising step, unmask the next k positions in the sequence (left-to-right).
    
    Unmasking with k=1 is equivalent to standard autoregressive generation.

    Returns:
        nll: [batch_size] negative log-likelihood averaged over sequence for each batch element
    """
    x = x[:, 0 : model.config.block_size].contiguous() #[batch_size, seq_len]
    batch_size, seq_len = x.shape
    mask_token_id = model.config.vocab_size
    
    z_t = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    log_probs = torch.zeros(batch_size, device=x.device) # [batch_size]

    steps = math.ceil(seq_len / k)
    for step in range(steps):
        # PREDICT: Get model probabilities
        logits = model(z_t) # [batch_size, seq_len, vocab_size]
        token_log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # SELECT: Identify next k positions to unmask (left-to-right)
        start_pos = step * k
        end_pos = min((step + 1) * k, seq_len)
        selected_positions = torch.arange(start_pos, end_pos, device=x.device).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, k]
        
        # UPDATE: Unmask next-k positions and accumulate log-probabilities
        for batch_idx in range(batch_size):
            for k_idx in range(end_pos - start_pos):
                pos = selected_positions[batch_idx, k_idx].item()
                true_token = x[batch_idx, pos].item()
                log_probs[batch_idx] += token_log_probs[batch_idx, pos, true_token]
                z_t[batch_idx, pos] = true_token
                
    nll = -log_probs / seq_len  # avg NLL per token
    return nll


def compute_deterministic_nll(
    x: torch.Tensor, model: nn.Module, strategy: str
) -> torch.Tensor:
    if strategy == "topk-confidence":
        nll = topk_confidence_nll(x, model)
    elif strategy == "block-topk-confidence":
        nll = block_topk_confidence_nll(x, model)
    elif strategy == "confidence threshold":
        nll = confidence_threshold_nll(x, model)
    elif strategy == "topk-margin":
        nll = topk_margin_nll(x, model)
    elif strategy == "autoregressive":
        nll = topk_autoregressive_nll(x, model)
    else:
        raise ValueError(f"Unknown unmasking strategy: {strategy}")
    return nll