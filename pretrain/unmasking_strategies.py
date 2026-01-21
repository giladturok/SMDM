"""Deterministic unmasking strategies for masked diffusion models allow for *exact* evaluation of negative log-likelihood of a data sample x. 

At each denoising step, a strategy defines 
(1) which *positions* to unmask
(2) how to determine the *value* the unmasked position takes from their vocabulary distribution
"""
import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _greedy_cheating_decoding_nll_step(
    model: torch.nn.Module, x: torch.Tensor, z_k: torch.Tensor, mask_token_id: int, k: int=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = x.shape
    is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
    
    logits = model(z_k) # [batch_size, seq_len, vocab_size]
    token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
    true_log_probs = torch.gather(
        token_log_probs, dim=-1, index=x.unsqueeze(-1)
    ).squeeze(-1) # [batch_size, seq_len]
    masked_log_probs = torch.where(
        is_masked, true_log_probs, torch.full_like(true_log_probs, float("-inf"))
    ) # [batch_size, seq_len]
    
    greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
    nll_greedy = -masked_log_probs[torch.arange(batch_size), greedy_position] # [batch_size]
    z_k = z_k.clone()  # Avoid in-place modification issues
    z_k[torch.arange(batch_size), greedy_position] = x[torch.arange(batch_size), greedy_position]
    
    return nll_greedy / seq_len, z_k


def greedy_cheating_decoding_nll(
    model: torch.nn.Module, x: torch.Tensor, block_size: int, mask_token_id: int
) -> torch.Tensor:
    x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = x.shape
    
    z_k = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    nll_greedy = torch.zeros(batch_size, device=x.device) # [batch_size]
    
    for num_unmasked in range(seq_len):
        nll_step, z_k = _greedy_cheating_decoding_nll_step(model, x, z_k, mask_token_id)
        nll_greedy += nll_step
    return nll_greedy


def _greedy_decoding_nll_step(
    model: torch.nn.Module, x: torch.Tensor, z_k: torch.Tensor, mask_token_id: int, k: int=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = x.shape
    is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
    
    logits = model(z_k) # [batch_size, seq_len, vocab_size]
    token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
    confidence_log_probs = torch.max(token_log_probs, dim=-1).values # [batch_size, seq_len]
    masked_log_probs = torch.where(
        is_masked, confidence_log_probs, torch.full_like(confidence_log_probs, float("-inf"))
    ) # [batch_size, seq_len]
    greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
    
    true_log_probs = torch.gather(
        token_log_probs, dim=-1, index=x.unsqueeze(-1)
    ).squeeze(-1) # [batch_size, seq_len]
    nll_greedy = -true_log_probs[torch.arange(batch_size), greedy_position] # [batch_size]
    z_k = z_k.clone()  # Avoid in-place modification issues
    z_k[torch.arange(batch_size), greedy_position] = x[torch.arange(batch_size), greedy_position]

    return nll_greedy / seq_len, z_k


def greedy_decoding_nll(
    model: torch.nn.Module, x: torch.Tensor, block_size: int, mask_token_id: int
) -> torch.Tensor:
    x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = x.shape
    
    z_k = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    nll_greedy = torch.zeros(batch_size, device=x.device) # [batch_size]
    
    for num_unmasked in range(seq_len):
        nll_step, z_k = _greedy_decoding_nll_step(model, x, z_k, mask_token_id)
        nll_greedy += nll_step
    return nll_greedy
        

def _greedy_block_decoding_nll_step(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    z_k: torch.Tensor, 
    block_start: int, 
    block_end: int,
    mask_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = x.shape
    is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
    seq_idx = torch.arange(seq_len, device=x.device)
    is_within_block = (seq_idx >= block_start) & (seq_idx < block_end)
    
    logits = model(z_k) # [batch_size, seq_len, vocab_size]
    token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
    confidence_log_probs = torch.max(token_log_probs, dim=-1).values # [batch_size, seq_len]
    bool_mask = is_masked & is_within_block
    masked_log_probs = torch.where(
        bool_mask, confidence_log_probs, torch.full_like(confidence_log_probs, float("-inf"))
    ) # [batch_size, seq_len]
    greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
    
    true_log_probs = torch.gather(
        token_log_probs, dim=-1, index=x.unsqueeze(-1)
    ).squeeze(-1) # [batch_size, seq_len]
    nll_greedy = -true_log_probs[torch.arange(batch_size), greedy_position] # [batch_size]
    z_k = z_k.clone()  # Avoid in-place modification issues
    z_k[torch.arange(batch_size), greedy_position] = x[torch.arange(batch_size), greedy_position]
    
    return nll_greedy / seq_len, z_k


def greedy_block_decoding_nll(
    model: torch.nn.Module, x: torch.Tensor, block_size: int, mask_token_id: int
) -> torch.Tensor:
    x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = x.shape
    z_k = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    nll_greedy = torch.zeros(batch_size, device=x.device) # [batch_size]
    
    num_blocks = math.ceil(seq_len / block_size)
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(seq_len, (block_idx + 1) * block_size)
        
        for num_unmasked in range(block_start, block_end):
            nll_step, z_k = _greedy_block_decoding_nll_step(
                model, x, z_k, block_start, block_end, mask_token_id
            )
            nll_greedy += nll_step
    return nll_greedy


def _uniform_decoding_nll_step(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    num_unmasked: int,
    mask_token_id: int,
    num_monte_carlo: int = 16
) -> torch.Tensor:
    batch_size, seq_len = x.shape
    num_masked = seq_len - num_unmasked
    
    nll_accumulator = torch.zeros(batch_size, device=x.device) # [batch_size]
    for _ in range(num_monte_carlo):
    
        random_permutations = torch.stack(
            [torch.randperm(seq_len, device=x.device) for _ in range(batch_size)]
        ) # [batch_size, seq_len]
        positions_to_mask = random_permutations[:, :num_masked] # [batch_size, num_masked]
        z_k = x.scatter(dim=-1, index=positions_to_mask, value=mask_token_id)
        
        logits = model(z_k) # [batch_size, seq_len, vocab_size]
        token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
        true_log_probs = torch.gather(
            token_log_probs, dim=-1, index=x.unsqueeze(-1)
        ).squeeze(-1) # [batch_size, seq_len]
        
        is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
        masked_log_probs = torch.where(
            is_masked, true_log_probs, torch.zeros_like(true_log_probs)
        ) # [batch_size, seq_len]
        nll_step = -masked_log_probs.sum(dim=-1) / num_masked # [batch_size]
        nll_accumulator += nll_step # [batch_size]
        
    nll_uniform = nll_accumulator / num_monte_carlo # [batch_size]
    return nll_uniform / seq_len
        

def uniform_decoding_nll(
    model: torch.nn.Module, x: torch.Tensor, block_size: int, mask_token_id: int
) -> torch.Tensor:
    x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = x.shape
    nll_uniform = torch.zeros(batch_size, device=x.device) # [batch_size]

    for num_unmasked in range(seq_len):
        nll_step = _uniform_decoding_nll_step(model, x, num_unmasked, mask_token_id)
        nll_uniform += nll_step
    return nll_uniform


def _probability_margin_decoding_nll_step(
    model: torch.nn.Module, x: torch.Tensor, z_k: torch.Tensor, mask_token_id: int, k: int=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = x.shape
    is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
    
    def _compute_probability_margin(token_log_probs):
        top_two_log_probs = torch.topk(token_log_probs, k=2, dim=-1).values # [batch_size, seq_len, 2]
        most_confident = top_two_log_probs[:, :, 0] # [batch_size, seq_len]
        second_most_confident = top_two_log_probs[:, :, 1] # [batch_size, seq_len]
        return most_confident - second_most_confident
    
    logits = model(z_k) # [batch_size, seq_len, vocab_size]
    token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
    margin_log_probs = _compute_probability_margin(token_log_probs) # [batch_size, seq_len]
    masked_log_probs = torch.where(
        is_masked, margin_log_probs, torch.full_like(margin_log_probs, float("-inf"))
    ) # [batch_size, seq_len]
    greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
    
    true_log_probs = torch.gather(
        token_log_probs, dim=-1, index=x.unsqueeze(-1)
    ).squeeze(-1) # [batch_size, seq_len]
    nll_greedy = -true_log_probs[torch.arange(batch_size), greedy_position] # [batch_size]
    z_k = z_k.clone()  # Avoid in-place modification issues
    z_k[torch.arange(batch_size), greedy_position] = x[torch.arange(batch_size), greedy_position]

    return nll_greedy / seq_len, z_k


def probability_margin_decoding_nll(
    model: torch.nn.Module, x: torch.Tensor, block_size: int, mask_token_id: int
) -> torch.Tensor:
    x = x[:, 0 : block_size].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = x.shape
    
    z_k = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    nll_margin = torch.zeros(batch_size, device=x.device) # [batch_size]
    
    for num_unmasked in range(seq_len):
        nll_step, z_k = _probability_margin_decoding_nll_step(model, x, z_k, mask_token_id)
        nll_margin += nll_step
    return nll_margin


def compute_deterministic_nll(
    x: torch.Tensor, 
    model: torch.nn.Module, 
    decoding_strategy: str,
    mask_token_id: int,
    block_size: int = 32,
) -> torch.Tensor:
    if decoding_strategy == "greedy":
        nll = greedy_decoding_nll(model, x, block_size, mask_token_id)
    elif decoding_strategy == "block-greedy":
        nll = greedy_block_decoding_nll(model, x, block_size, mask_token_id)
    elif decoding_strategy == "uniform":
        nll = uniform_decoding_nll(model, x, block_size, mask_token_id)
    elif decoding_strategy == "probability-margin":
        nll = probability_margin_decoding_nll(model, x, block_size, mask_token_id)
    elif decoding_strategy == "greedy-cheating":
        nll = greedy_cheating_decoding_nll(model, x, block_size, mask_token_id)
    else:
        raise ValueError(f"Unknown unmasking strategy: {decoding_strategy}")
    return nll