import glob
import math
import os
os.environ["LIGHTNING_UPGRADE_CHECK"] = "0"
os.environ["LIGHTNING_JUPYTER_MODE"] = "0"
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
import torch.nn.functional as F
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.diffmodel import TransEncoder, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import random
import argparse
from unmasking_strategies import compute_deterministic_nll


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parse.add_argument('--flops', type=float, help='FLOPs, *e18')
    parse.add_argument('--batch_size', type=int, default=256, help='global_batch_size')
    args = parse.parse_args()
    return args

args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'  # config
out_dir = Path('workdir')

model_para_config = {
    '6': 6.294784, '19': 18.880896, '34': 33.563136, '48': 47.786688, '66': 65.54944,
    '85': 85.21408, '75': 75.38752, '113': 113.265408, '142': 141.581568, '170': 169.897728,
    '180': 179.856768, '206': 205.550464, '231': 231.24416, '268': 268.469248, '302': 302.027776,
    '336': 335.586304, '472': 471.90656, '551': 550.55744, '571': 571.001728, '629': 629.20832,
    '666': 666.168448, '717': 717.285888, '761': 761.335168, '831': 830.541312, '944': 943.796736,
    '1028': 1027.677952, '1233': 1233.213184, '1476': 1476.487168, '1678': 1677.826048, '2121': 2121.39328
}

# Hyperparameters
num_of_devices = 1 # prev 8
global_batch_size = int(args.batch_size / args.nodes_num)
learning_rate = 2e-4
if args.model <= 20:
    micro_batch_size = 32
elif args.model <= 50:
    micro_batch_size = 16
elif args.model <= 1000:
    micro_batch_size = 8
else:
    micro_batch_size = 4
max_step = int(args.flops * 1e12 / (6 * model_para_config[f'{args.model}'] * global_batch_size * 2048) / args.nodes_num)
warmup_steps = int(max_step / 100) if int(max_step / 100) > 100 else 100
log_step_interval = 10
eval_iters = int(100 * 1024 / global_batch_size)
save_step_interval = 5000
eval_step_interval = 999999999999 #inf


weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 2e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps




max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
train_data_config = [
    ("train_slim", 1.0),
    ("train_star", 0.),
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


def setup(
    devices: int = 1, # prev 8
    train_data_dir: Path = Path("/dataset/slim_star_combined"),
    val_data_dir: Path = Path("./data/slim_pajama/tokenized"),
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'mdm-{args.model}M-{int(args.flops)}e18'
    # out_dir = Path('workdir/scaling_debug') / hp_name
    out_dir = Path("models/mdm_safetensors") / hp_name
    print(f"Outputs will be saved to {out_dir}")
    wandb_logger = WandbLogger(name=f'{hp_name}-mc', save_dir=out_dir, project='scaling')

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    
    effective_block_size = config.block_size + 1
    val_dataloader = create_dataloader(
        batch_size=2, # micro_batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=val_data_dir,
        shuffle=False,
        seed=3407,
        split="validation"
    )

    fabric.setup_dataloaders(val_dataloader)
    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))
 

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}
    
    checkpoint_path = out_dir / f"mdm-{args.model}M-{int(args.flops)}e18.safetensors"

    if checkpoint_path.exists():
        fabric.print(f"Loading checkpoint from {checkpoint_path}")
        # fabric.load(checkpoint_path, state, strict=False)
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    validate_time = time.perf_counter()
    # train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    validate(fabric, model, val_dataloader)
    fabric.print(f"Validation time: {(time.perf_counter()-validate_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters: int = 100) -> dict:
    """Evaluate model using multiple NLL estimation strategies.
    
    Returns:
        dict: NLL estimates for each strategy
    """
    fabric.print("Validating ...")
    model.eval()
    
    # Initialize accumulators
    # strategies = ["mc", "topk", "block_topk", "topk_margin", "confidence_thresh", "autoregressive"]
    strategies = ["mc", "nll_greedy", "nelbo_uniform"]
    accumulators = {s: torch.zeros(1, device=fabric.device) for s in strategies}
    counter = torch.zeros(1, device=fabric.device)
    
    for idx, val_data in enumerate(val_dataloader):
        if idx >= eval_iters:
            break
        
        val_data = val_data.to(fabric.device)
        batch_size = val_data.shape[0]
        counter += batch_size
        
        nll_greedy, nelbo_uniform = greedy_and_uniform_decoding(model, val_data)
        
        # Compute NLLs for each strategy
        losses = {
            "mc": _validate_mc(fabric, val_data, model),
            "nll_greedy": nll_greedy,
            "nelbo_uniform": nelbo_uniform,
            # "topk": compute_deterministic_nll(val_data, model, strategy="topk-confidence"),
            # "block_topk": compute_deterministic_nll(val_data, model, strategy="block-topk-confidence"),
            # "topk_margin": torch.tensor([1]), # compute_deterministic_nll(val_data, model, strategy="topk-margin"),
            # "confidence_thresh": torch.tensor([1]), # compute_deterministic_nll(val_data, model, strategy="confidence threshold"),
            # "autoregressive": torch.tensor([1]), # compute_deterministic_nll(val_data, model, strategy="autoregressive"),
        }
        
        # Accumulate
        for strategy, loss_tensor in losses.items():
            accumulators[strategy] += loss_tensor.sum().item()
        
        # Print running averages
        running_avgs = {s: (accumulators[s] / counter).item() for s in strategies}
        fabric.print(
            f"Val iter {idx:04}:\t"
            f"NLL Bound {running_avgs['mc']:.4f},\t"
            f"NELBO Uniform {running_avgs['nelbo_uniform']:.4f},\t"
            f"NLL Greedy {running_avgs['nll_greedy']:.4f}\t"
            # f"Top-K {running_avgs['topk']:.4f},\t"
            # f"Conf. Thresh {running_avgs['confidence_thresh']:.4f},\t"
            # f"Top-K Margin {running_avgs['topk_margin']:.4f},\t"
            # f"Block Top-K {running_avgs['block_topk']:.4f},\t"
            # f"Autoregressive {running_avgs['autoregressive']:.4f},\t"
            f"Counted Samples {counter.item()}"
        )
    
    # Synchronize across devices
    total_counter = fabric.all_reduce(counter, reduce_op="sum").item()
    final_results = {}
    
    for strategy in strategies:
        avg_loss = fabric.all_reduce(accumulators[strategy], reduce_op="sum") / total_counter
        final_results[strategy] = avg_loss.item()
    
    # Print final results
    if fabric.global_rank == 0:
        fabric.print("\n" + "="*60)
        fabric.print("FINAL VALIDATION RESULTS:")
        fabric.print(f"  NLL Bound (MC):          {final_results['mc']:.4f}")
        fabric.print(f"  NELBO Uniform            {final_results['nll_greedy']:.4f}")
        fabric.print(f"  NLL Greedy               {final_results['nelbo_uniform']:.4f}")
        # fabric.print(f"  NLL Top-K:               {final_results['topk']:.4f}")
        # fabric.print(f"  NLL Confidence Thresh:   {final_results['confidence_thresh']:.4f}")
        # fabric.print(f"  NLL Top-K Margin:        {final_results['topk_margin']:.4f}")
        # fabric.print(f"  NLL Block Top-K:         {final_results['block_topk']:.4f}")
        # fabric.print(f"  NLL Autoregressive:      {final_results['autoregressive']:.4f}")
        fabric.print("="*60 + "\n")
    
    model.train()
    return final_results


@torch.no_grad()
def validate_old(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    counter = torch.zeros(1, device=fabric.device)
    losses_mc_accum = torch.zeros(1, device=fabric.device)
    losses_topk_accum = torch.zeros(1, device=fabric.device)
    losses_block_topk_accum = torch.zeros(1, device=fabric.device)
    losses_confidence_thresh_accum = torch.zeros(1, device=fabric.device)
    losses_topk_margin_accum = torch.zeros(1, device=fabric.device)
    losses_autoregressive_accum = torch.zeros(1, device=fabric.device)
    
    # losses_mc = torch.zeros(eval_iters, device=fabric.device)
    # losses_topk = torch.zeros(eval_iters, device=fabric.device)
    # losses_confidence_thresh = torch.zeros(eval_iters, device=fabric.device)
    # losses_topk_margin = torch.zeros(eval_iters, device=fabric.device)
    # losses_block_topk = torch.zeros(eval_iters, device=fabric.device)
    # losses_autoregressive = torch.zeros(eval_iters, device=fabric.device)
    eval_iters = 2
    for idx, val_data in enumerate(val_dataloader):
        if idx >= eval_iters:
            break
        
        val_data = val_data.to(fabric.device)
        counter += val_data.shape[0] # total batch size processed
        
        losses_mc = _validate_mc(fabric, val_data, model)
        losses_mc_accum += losses_mc.sum().item()
        
        losses_topk = compute_deterministic_nll(val_data, model, strategy="topk-confidence")
        losses_topk_accum += losses_topk.sum().item()
        
        losses_block_topk = compute_deterministic_nll(val_data, model, strategy="block-topk-confidence")
        losses_block_topk_accum += losses_block_topk.sum().item()
        
        losses_topk_margin = compute_deterministic_nll(val_data, model, strategy="topk-margin")
        losses_topk_margin_accum += losses_topk_margin.sum().item()

        losses_confidence_thresh = compute_deterministic_nll(val_data, model, strategy="confidence threshold")
        losses_confidence_thresh_accum += losses_confidence_thresh.sum().item()
        
        losses_autoregressive = compute_deterministic_nll(val_data, model, strategy="autoregressive")
        losses_autoregressive_accum += losses_autoregressive.sum().item()
        
        # print the accumulated losses divided by the counter
        print(f"Val iter {idx}:\tNLL Bound {losses_mc_accum.item() / counter.item():.4f},\tNLL Top-K {losses_topk_accum.item() / counter.item():.4f},\tNLL Confidence Threshold {losses_confidence_thresh_accum.item() / counter.item():.4f},\tNLL Top-K Margin {losses_topk_margin_accum.item() / counter.item():.4f},\tNLL Block Top-K {losses_block_topk_accum.item() / counter.item():.4f},\tNLL Autoregressive {losses_autoregressive_accum.item() / counter.item():.4f}\tCounted Samples {counter.item()}")
        # print(f"Val iter {idx}:\tNLL Bound {losses_mc[idx].item():.4f},\tNLL Top-K {losses_topk[idx].item():.4f},\tNLL Confidence Threshold {losses_confidence_thresh[idx].item():.4f},\tNLL Top-K Margin {losses_topk_margin[idx].item():.4f},\tNLL Block Top-K {losses_block_topk[idx].item():.4f},\tNLL Autoregressive {losses_autoregressive[idx].item():.4f}")

    total_counter = fabric.all_reduce(torch.tensor(counter, device=fabric.device), reduce_op="sum").item()
    
    losses_mc = fabric.all_reduce(torch.tensor(losses_mc_accum, device=fabric.device), reduce_op="sum") / total_counter
    
    losses_topk = fabric.all_reduce(torch.tensor(losses_topk_accum, device=fabric.device), reduce_op="sum") / total_counter
    
    losses_block_topk = fabric.all_reduce(torch.tensor(losses_block_topk_accum, device=fabric.device), reduce_op="sum") / total_counter
    
    losses_confidence_thresh = fabric.all_reduce(torch.tensor(losses_confidence_thresh_accum, device=fabric.device), reduce_op="sum") / total_counter
    
    losses_topk_margin = fabric.all_reduce(torch.tensor(losses_topk_margin_accum, device=fabric.device), reduce_op="sum") / total_counter
    
    losses_autoregressive = fabric.all_reduce(torch.tensor(losses_autoregressive_accum, device=fabric.device), reduce_op="sum") / total_counter
    
    if fabric.global_rank == 0:
        print(f"Final Validation NLL Bound: {losses_mc.item():.4f}")
        print(f"Final Validation NLL Top-K: {losses_topk.item():.4f}")
        print(f"Final Validation NLL Confidence Threshold: {losses_confidence_thresh.item():.4f}")
        print(f"Final Validation NLL Top-K Margin: {losses_topk_margin.item():.4f}")
        print(f"Final Validation NLL Block Top-K: {losses_block_topk.item():.4f}")
        print(f"Final Validation NLL Autoregressive: {losses_autoregressive.item():.4f}")

    model.train()
    return losses_mc


def _validate_mc_old(
    fabric: L.Fabric, val_data: torch.Tensor, model: torch.nn.Module
) -> torch.Tensor:
    """Upper bound on negative log-likelihood via Monte Carlo estimation."""
    mc_num = 128
    mc_loss = torch.zeros(mc_num, device=val_data.device)
    for i in range(mc_num):
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        noisy_input, mask_indices, p_mask = forward_process(input_ids)
        logits = model(noisy_input) #[batch_size, seq_len, vocab_size]
        loss = F.cross_entropy(
            logits[mask_indices], input_ids[mask_indices], reduction='none'
        ) / p_mask[mask_indices] #[num_masked_positions]
        print(f"Loss shape: {loss.shape}, Input shape: {input_ids.shape}, Mask shape: {mask_indices.shape}, P_mask shape: {p_mask.shape}")
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1]) # normalized by total number of tokens
        print(f"Loss shape after sum: {loss.shape}")
        mc_loss[i] = loss
    return mc_loss.mean().item()


def greedy_and_uniform_decoding(
    model: torch.nn.Module, x: torch.Tensor, num_monte_carlo: int = 16
):
    x = x[:, 0 : model.config.block_size].contiguous() #[batch_size, seq_len]
    batch_size, seq_len = x.shape
    mask_token_id = model.config.vocab_size
    nll_greedy_list, nelbo_uniform_list = [], []
    
    z_k = torch.full_like(x, mask_token_id, device=x.device) # [batch_size, seq_len]
    
    # Iterate over number of un-masked tokens k
    for k in range(seq_len):
        
        # GREEDY: Compute exact NLL under greedy decoding
        logits = model(z_k) # [batch_size, seq_len, vocab_size]
        token_log_probs = F.log_softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
        true_log_probs = torch.gather(
            token_log_probs, dim=-1, index=x.unsqueeze(-1)
        ).squeeze(-1) # [batch_size, seq_len]
        
        is_masked = (z_k == mask_token_id) # [batch_size, seq_len]
        masked_log_probs = torch.where(
            is_masked, true_log_probs, torch.full_like(true_log_probs, float("-inf"))
        ) # [batch_size, seq_len]
        greedy_position = masked_log_probs.argmax(dim=-1) # [batch_size]
        nll_k_greedy = -masked_log_probs[torch.arange(batch_size), greedy_position] # [batch_size]
        z_k[torch.arange(batch_size), greedy_position] = x[torch.arange(batch_size), greedy_position] # [batch_size, seq_len]
        nll_greedy_list.append(nll_k_greedy)
        
        # UNIFORM: Compute a NLL bound under uniform decoding aka NELBO
        total_samples = batch_size * num_monte_carlo
        random_permutations = torch.stack([
            torch.randperm(seq_len, device=x.device) for _ in range(total_samples)
        ]) # [batch_size * num_monte_carlo, seq_len]
        num_masked = seq_len - k
        positions_to_mask = random_permutations[:, :num_masked] # [batch_size * num_monte_carlo, num_masked]
        
        x_all = x.repeat_interleave(num_monte_carlo, dim=0) # [batch_size * num_monte_carlo, seq_len]
        z_k_all = x_all.clone() # [batch_size * num_monte_carlo, seq_len]
        z_k_all = z_k_all.scatter(dim=1, index=positions_to_mask, value=mask_token_id)
        logits_all = model(z_k_all) # [batch_size * num_monte_carlo, seq_len, vocab_size]
        token_log_probs_all = F.log_softmax(logits_all, dim=-1) # [batch_size * num_monte_carlo, seq_len, vocab_size]
        true_log_probs_all = torch.gather(
            token_log_probs_all, dim=-1, index=x_all.unsqueeze(-1)
        ).squeeze(-1) # [batch_size * num_monte_carlo, seq_len]
        
        is_masked_all = (z_k_all == mask_token_id)
        masked_log_probs_all = torch.where(
            is_masked_all, true_log_probs_all, torch.zeros_like(true_log_probs_all)
        ) # [batch_size * num_monte_carlo, seq_len]
        nelbo_all_samples = - masked_log_probs_all.sum(dim=-1) / num_masked # [batch_size * num_monte_carlo]
        
        nelbo_samples = nelbo_all_samples.view(batch_size, num_monte_carlo) # [batch_size, num_monte_caro]
        nelbo_k_uniform = nelbo_samples.mean(dim=1) # [batch_size]
        nelbo_uniform_list.append(nelbo_k_uniform)
        
        print(f"Num unmasked tokens {k}:\tNLL Greedy: {nll_k_greedy}\tNELBO Uniform: {nelbo_k_uniform}")
    
    nll_greedy = torch.stack(nll_greedy_list).sum(dim=0) # [batch_size]
    nelbo_uniform = torch.stack(nelbo_uniform_list).sum(dim=0) # [batch_size]
    nll_greedy_per_token = nll_greedy / seq_len
    nelbo_uniform_per_token = nelbo_uniform / seq_len
    print(f"NLL Greedy: {nll_greedy_per_token}\tNELBO Uniform: {nelbo_uniform_per_token}")
    return nll_greedy_per_token, nelbo_uniform_per_token

# === TEST HARNESS (add to both files) ===

def test_nelbo_equivalence(model, x, t_frac=0.9, eps=1e-3):
    """
    Compare NELBO computation between methods using identical mask.
    
    Args:
        t_frac: timestep as fraction in [0,1] (e.g., 0.5 for middle of diffusion)
    """
    x = x[:, 0 : model.config.block_size].contiguous() #[batch_size, seq_len]
    batch_size, seq_len = x.shape
    mask_token_id = model.config.vocab_size
    
    # Generate shared mask
    p_mask_val = (1 - eps) * t_frac + eps
    p_mask = torch.full((batch_size, seq_len), p_mask_val, device=x.device)
    mask_indices = torch.rand((batch_size, seq_len), device=x.device) < p_mask
    
    # Create masked input
    z_t = torch.where(mask_indices, mask_token_id, x)
    
    # Get logits
    logits = model(z_t)
    
    # ===== METHOD 1 STYLE =====
    token_losses_1 = F.cross_entropy(
        logits[mask_indices], 
        x[mask_indices], 
        reduction='none'
    ) / p_mask[mask_indices]
    
    batch_ids_masked = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(mask_indices)[mask_indices]
    batch_losses_1 = torch.zeros(batch_size, device=x.device)
    batch_losses_1.scatter_add_(0, batch_ids_masked, token_losses_1)
    nelbo_1 = batch_losses_1 / seq_len
    
    # ===== METHOD 2 STYLE =====
    token_log_probs = torch.log_softmax(logits, dim=-1)
    log_probs_true = token_log_probs.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
    log_probs_masked = torch.where(mask_indices, log_probs_true, torch.zeros_like(log_probs_true))
    importance_weight = p_mask_val * seq_len
    nelbo_2 = -log_probs_masked.sum(dim=-1) / importance_weight
    
    # ===== COMPARISON =====
    print(f"\n=== NELBO Equivalence Test (t={t_frac:.3f}) ===")
    print(f"Method 1 (MC style):    {nelbo_1.cpu().numpy()}")
    print(f"Method 2 (greedy style): {nelbo_2.cpu().numpy()}")
    print(f"Difference:              {(nelbo_1 - nelbo_2).abs().cpu().numpy()}")
    print(f"Max difference:          {(nelbo_1 - nelbo_2).abs().max().item():.6f}")
    print(f"Num masked per sample:   {mask_indices.sum(dim=1).cpu().numpy()}")
    
    return nelbo_1, nelbo_2


def _validate_mc(
    fabric: L.Fabric, val_data: torch.Tensor, model: torch.nn.Module, mc_num: int = 128
) -> torch.Tensor:
    """Upper bound on negative log-likelihood via Monte Carlo estimation.
    
    Returns:
        nll [batch_size]: NLL upper bound per batch element, averaged over sequence length
    """
    input_ids = val_data[:, :model.config.block_size].contiguous() #[batch_size, seq_len]
    batch_size, seq_len = input_ids.shape
    
    # Accumulator for MC samples: [mc_num, batch_size]
    mc_losses = torch.zeros(mc_num, batch_size, device=val_data.device)
    
    for i in range(mc_num):
        noisy_input, mask_indices, p_mask = forward_process(input_ids)
        logits = model(noisy_input)  # [batch_size, seq_len, vocab_size]
        
        # Compute per-token losses: [num_masked_positions]
        token_losses = F.cross_entropy(
            logits[mask_indices], # [batch_size, num_masked, vocab_size]
            input_ids[mask_indices], # [batch_size, seq_len]
            reduction='none'
        ) / p_mask[mask_indices]
        
        # Accumulate losses per batch element
        # Get batch indices for each masked position
        batch_indices = torch.arange(batch_size, device=val_data.device).unsqueeze(1).expand_as(mask_indices)
        batch_ids_masked = batch_indices[mask_indices]  # [num_masked_positions]
        
        # Scatter-add losses to corresponding batch
        batch_losses = torch.zeros(batch_size, device=val_data.device)
        batch_losses.scatter_add_(0, batch_ids_masked, token_losses)
        
        # Normalize by sequence length
        mc_losses[i] = batch_losses / seq_len
    
    # Average over MC samples: [batch_size]
    nll = mc_losses.mean(dim=0)
    return nll


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)
        print(f"Creating dataset from {len(filenames)} files at {data_dir} with prefix {prefix} for split {split}")

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    # train_dataloader = create_dataloader(
    #     batch_size=batch_size,
    #     block_size=effective_block_size,
    #     fabric=fabric,
    #     data_dir=train_data_dir,
    #     shuffle=True,
    #     seed=seed,
    #     split="train"
    # )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    # return train_dataloader, val_dataloader
    return val_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup()
