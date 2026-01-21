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
from lit_gpt.utils import (
    chunked_cross_entropy,
    get_default_supported_precision,
    num_parameters,
    step_csv_logger,
    lazy_load,
)
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import random
import argparse
from unmasking_strategies import compute_deterministic_nll


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=int, help="model parameters")
    parse.add_argument("--nodes_num", type=int, default=1, help="number of nodes")
    parse.add_argument("--flops", type=float, help="FLOPs, *e18")
    parse.add_argument("--batch_size", type=int, default=256, help="global_batch_size")
    args = parse.parse_args()
    return args


args = parse_args()
model_name = f"Diff_LLaMA_{args.model}M"  # config
out_dir = Path("workdir")

model_para_config = {
    "6": 6.294784,
    "19": 18.880896,
    "34": 33.563136,
    "48": 47.786688,
    "66": 65.54944,
    "85": 85.21408,
    "75": 75.38752,
    "113": 113.265408,
    "142": 141.581568,
    "170": 169.897728,
    "180": 179.856768,
    "206": 205.550464,
    "231": 231.24416,
    "268": 268.469248,
    "302": 302.027776,
    "336": 335.586304,
    "472": 471.90656,
    "551": 550.55744,
    "571": 571.001728,
    "629": 629.20832,
    "666": 666.168448,
    "717": 717.285888,
    "761": 761.335168,
    "831": 830.541312,
    "944": 943.796736,
    "1028": 1027.677952,
    "1233": 1233.213184,
    "1476": 1476.487168,
    "1678": 1677.826048,
    "2121": 2121.39328,
}

# Hyperparameters
num_of_devices = 1  # prev 8
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
max_step = int(
    args.flops
    * 1e12
    / (6 * model_para_config[f"{args.model}"] * global_batch_size * 2048)
    / args.nodes_num
)
warmup_steps = int(max_step / 100) if int(max_step / 100) > 100 else 100
log_step_interval = 10
eval_iters = int(100 * 1024 / global_batch_size)
save_step_interval = 5000
eval_step_interval = 999999999999  # inf


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
    ("train_star", 0.0),
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}
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
    devices: int = 1,  # prev 8
    train_data_dir: Path = Path("/dataset/slim_star_combined"),
    val_data_dir: Path = Path("./data/slim_pajama/tokenized"),
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f"mdm-{args.model}M-{int(args.flops)}e18"
    # out_dir = Path('workdir/scaling_debug') / hp_name
    out_dir = Path("models/mdm_safetensors") / hp_name
    print(f"Outputs will be saved to {out_dir}")
    wandb_logger = WandbLogger(
        name=f"{hp_name}-mc", save_dir=out_dir, project="scaling"
    )

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

    fabric = L.Fabric(
        devices=devices,
        strategy=strategy,
        precision=precision,
        loggers=[logger, wandb_logger],
    )
    fabric.print(hparams)
    # fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(
        fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval
    )

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    effective_block_size = config.block_size + 1
    val_dataloader = create_dataloader(
        batch_size=8,  # micro_batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=val_data_dir,
        shuffle=False,
        seed=3407,
        split="validation",
    )

    fabric.setup_dataloaders(val_dataloader)
    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0,
    }

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
def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    eval_iters: int = 100,
) -> dict:
    """Evaluate model using multiple NLL estimation strategies.

    Returns:
        dict: NLL estimates for each strategy
    """
    fabric.print("Validating ...")
    model.eval()

    # Initialize accumulators
    strategies = ["nll_uniform", "nll_greedy", "nll_block_greedy", "nll_probability_margin", "nll_greedy_cheating"]
    accumulators = {s: torch.zeros(1, device=fabric.device) for s in strategies}
    counter = torch.zeros(1, device=fabric.device)

    for idx, val_data in enumerate(val_dataloader):
        if idx >= eval_iters:
            break

        val_data = val_data.to(fabric.device)
        batch_size = val_data.shape[0]
        counter += batch_size

        # nll_greedy, nelbo_uniform = greedy_and_uniform_decoding(model, val_data)

        # Compute NLLs for each strategy
        losses = {
            "nll_greedy": compute_deterministic_nll(val_data, model, strategy="greedy"),
            # "mc": _validate_mc(fabric, val_data, model),
            "nll_uniform": compute_deterministic_nll(val_data, model, strategy="uniform"),
            "nll_block_greedy": compute_deterministic_nll(val_data, model, strategy="block-greedy"),
            "nll_probability_margin": compute_deterministic_nll(val_data, model, strategy="probability-margin"),
            "nll_greedy_cheating": compute_deterministic_nll(val_data, model, strategy="greedy-cheating")
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
            # f"NLL Bound {running_avgs['mc']:.4f},\t"
            f"NLL Uniform {running_avgs['nll_uniform']:.4f},\t"
            f"NLL Greedy {running_avgs['nll_greedy']:.4f}\t"
            f"NLL Block Greedy {running_avgs['nll_block_greedy']:.4f}\t"
            f"NLL Probability Margin {running_avgs['nll_probability_margin']:.4f}\t"
            f"NLL Greedy Cheating {running_avgs['nll_greedy_cheating']:.4f}\t"
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
        avg_loss = (
            fabric.all_reduce(accumulators[strategy], reduce_op="sum") / total_counter
        )
        final_results[strategy] = avg_loss.item()

    # Print final results
    if fabric.global_rank == 0:
        fabric.print("\n" + "=" * 60)
        fabric.print("FINAL VALIDATION RESULTS:")
        # fabric.print(f"  NLL Bound (MC):          {final_results['mc']:.4f}")
        fabric.print(f"  NLL Uniform               {final_results['nll_uniform']:.4f}")
        fabric.print(f"  NLL Greedy            {final_results['nll_greedy']:.4f}")
        fabric.print(f"  NLL Block-Greedy            {final_results['nll_block_greedy']:.4f}")
        fabric.print(f"  NLL Probability Margin        {final_results['nll_probability_margin']:.4f}")
        fabric.print(f"   NLL Greedy Cheating            {final_results['nll_greedy_cheating']:.4f}")
        # fabric.print(f"  NLL Top-K:               {final_results['topk']:.4f}")
        # fabric.print(f"  NLL Confidence Thresh:   {final_results['confidence_thresh']:.4f}")
        # fabric.print(f"  NLL Top-K Margin:        {final_results['topk_margin']:.4f}")
        # fabric.print(f"  NLL Block Top-K:         {final_results['block_topk']:.4f}")
        # fabric.print(f"  NLL Autoregressive:      {final_results['autoregressive']:.4f}")
        fabric.print("=" * 60 + "\n")

    model.train()
    return final_results


def _validate_mc(
    fabric: L.Fabric, val_data: torch.Tensor, model: torch.nn.Module, mc_num: int = 128
) -> torch.Tensor:
    """Upper bound on negative log-likelihood via Monte Carlo estimation.

    Returns:
        nll [batch_size]: NLL upper bound per batch element, averaged over sequence length
    """
    input_ids = val_data[
        :, : model.config.block_size
    ].contiguous()  # [batch_size, seq_len]
    batch_size, seq_len = input_ids.shape

    # Accumulator for MC samples: [mc_num, batch_size]
    mc_losses = torch.zeros(mc_num, batch_size, device=val_data.device)

    for i in range(mc_num):
        noisy_input, mask_indices, p_mask = forward_process(input_ids)
        logits = model(noisy_input)  # [batch_size, seq_len, vocab_size]

        # Compute per-token losses: [num_masked_positions]
        token_losses = (
            F.cross_entropy(
                logits[mask_indices],  # [batch_size, num_masked, vocab_size]
                input_ids[mask_indices],  # [batch_size, seq_len]
                reduction="none",
            )
            / p_mask[mask_indices]
        )

        # Accumulate losses per batch element
        # Get batch indices for each masked position
        batch_indices = (
            torch.arange(batch_size, device=val_data.device)
            .unsqueeze(1)
            .expand_as(mask_indices)
        )
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
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
    split="train",
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)
        print(
            f"Creating dataset from {len(filenames)} files at {data_dir} with prefix {prefix} for split {split}"
        )

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size.
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
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

    return DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )


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
            split="validation",
        )
        if val_data_dir
        else None
    )
    # return train_dataloader, val_dataloader
    return val_dataloader, val_dataloader


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup()
