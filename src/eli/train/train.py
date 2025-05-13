import gc
import json
import os
from typing import Tuple
import datetime

import boto3
import torch
import torch.distributed as dist
from dacite import from_dict
from jaxtyping import Int
from torch import Tensor
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from eli.datasets.config import DatasetConfig
from eli.train.config import TrainConfig, encoder_cfg, train_cfg
from eli.train.download import download_dataset
from eli.train.encoder import EncoderDecoder, get_loss, get_loss_control


def preprocess_target_generated_tokens(
    target_generated_tokens: Int[Tensor, "batch tok"], tokenizer: AutoTokenizer
) -> Tuple[Int[Tensor, "batch tok"], Int[Tensor, "batch tok"]]:
    decoded = tokenizer.batch_decode(
        sequences=target_generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    decoded = [r.strip() for r in decoded]
    encoded = tokenizer(
        decoded, add_special_tokens=False, return_tensors="pt", padding=True
    )
    target_generated_tokens = encoded.input_ids
    attention_mask = encoded.attention_mask
    return target_generated_tokens, attention_mask


def save_encoder(encoder_decoder: EncoderDecoder, save_path: str):
    """Save just the encoder part of the encoder-decoder model.

    Args:
        save_path: Path where the model should be saved
    """
    # Get the encoder, accounting for DataParallel
    encoder = encoder_decoder.encoder

    # Move to CPU before saving
    encoder = encoder.cpu()

    # Save the model
    try:
        torch.save(encoder.state_dict(), save_path)
        print(f"Encoder saved to {save_path}")
    except Exception as e:
        print(f"Failed to save encoder: {e}")


def get_gradient_stats(parameters):
    """Calculate gradient statistics for the given parameters.

    Args:
        parameters: Iterator over parameters with gradients

    Returns:
        Dictionary containing gradient norm, absolute max, and absolute min
    """
    # Collect all gradients
    grads = [p.grad.detach() for p in parameters if p.grad is not None]

    if not grads:
        return {
            "grad_norm": torch.tensor(0.0),
            "grad_abs_max": torch.tensor(0.0),
            "grad_abs_min": torch.tensor(0.0),
        }

    # Flatten gradients
    flat_grads = torch.cat([g.flatten() for g in grads])

    # Calculate statistics
    grad_norm = torch.norm(flat_grads, p=2)
    grad_abs_max = torch.max(torch.abs(flat_grads))

    # Handle case where all gradients might be zero
    non_zero_grads = (
        flat_grads[flat_grads != 0] if torch.any(flat_grads != 0) else flat_grads
    )
    grad_abs_min = torch.min(torch.abs(non_zero_grads))

    return {
        "grad_norm": grad_norm,
        "grad_abs_max": grad_abs_max,
        "grad_abs_min": grad_abs_min,
    }


def init_distributed():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Initialize the process group with NCCL backend for GPU
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Initialize with gloo backend for CPU
        dist.init_process_group(backend="gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cpu")
    
    # Get world size and rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return world_size, rank, local_rank, device


def pull_dataset_config(train_cfg: TrainConfig) -> DatasetConfig:
    """
    Downloads the dataset configuration from S3 and returns a DatasetConfig object.
    Uses dacite for proper dataclass deserialization.

    Returns:
        DatasetConfig: The dataset configuration
    """

    # Create S3 client
    s3_client = boto3.client("s3")

    # Download config from S3
    dataset_name = train_cfg.dataset_name
    s3_key = f"datasets/{dataset_name}/config.json"

    response = s3_client.get_object(Bucket=train_cfg.s3_bucket, Key=s3_key)
    config_json = response["Body"].read().decode("utf-8")
    config_dict = json.loads(config_json)

    # Use dacite to properly deserialize the dict into a dataclass
    dataset_config = from_dict(data_class=DatasetConfig, data=config_dict)

    print(f"Downloaded dataset configuration from s3://{train_cfg.s3_bucket}/{s3_key}")
    return dataset_config


def train():
    # ----- Setup -----
    world_size, rank, local_rank, device = init_distributed()

    # Get dataset config from S3
    dataset_cfg = pull_dataset_config(train_cfg)
    assert (
        dataset_cfg.num_samples >= train_cfg.num_samples
    ), "Must have at least as many samples as requested"

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.decoder_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    encoder_decoder = EncoderDecoder(tokenizer, dataset_cfg, encoder_cfg, train_cfg).to(device)
    assert encoder_decoder.encoder.get_device() == device
    assert encoder_decoder.decoder.device == device

    # Set up DDP for both CPU and GPU
    if device.type == "cuda":
        encoder_decoder_ddp = DDP(encoder_decoder, device_ids=[local_rank])
    else:
        # For CPU, don't specify device_ids
        encoder_decoder_ddp = DDP(encoder_decoder)
        
    optimizer = torch.optim.AdamW(
        [param for param in encoder_decoder.parameters() if param.requires_grad],
        lr=encoder_cfg.lr,
        betas=encoder_cfg.betas,
        weight_decay=encoder_cfg.weight_decay,
    )

    # Initialize data loader
    data_loader = iter(download_dataset(train_cfg))

    # Set up gradient scaler for mixed precision
    loss_scaler = GradScaler()

    # Calculate number of training iterations
    combined_batch_size = train_cfg.dataset_loader_batch_size * world_size
    num_batches = train_cfg.num_samples // combined_batch_size

    # Initialize Wandb
    if train_cfg.wandb_enabled:
        run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{train_cfg.decoder_model_name.split('/')[-1]}"
        wandb.init(project="eli", name=run_name)
        wandb.config.update(train_cfg)
        wandb.config.update(encoder_cfg)
        wandb.config.update(dataset_cfg)

    # ----- Training loop -----
    try:
        for train_iter in tqdm(range(num_batches), desc="Training"):
            # Load data
            target_acts, target_generated_tokens = next(data_loader)
            assert target_acts.shape[0] == target_generated_tokens.shape[0] == train_cfg.dataset_loader_batch_size
            target_acts, target_generated_tokens = (
                target_acts.to(device),
                target_generated_tokens.to(device),
            )

            # Preprocess data
            target_generated_tokens, attention_mask = (
                preprocess_target_generated_tokens(target_generated_tokens, tokenizer)
            )

            # Update model
            optimizer.zero_grad()

            loss = get_loss(
                train_cfg,
                device,
                encoder_decoder_ddp,
                target_generated_tokens,
                attention_mask,
                target_acts,
                tokenizer,
                train_iter,
            )

            # Backward pass with loss scaling
            loss_scaler.scale(loss).backward()
            loss_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(encoder_decoder.parameters(), max_norm=5.0)

            loss_scaler.step(optimizer)
            loss_scaler.update()

            # ----- Logging -----
            if rank == 0 and train_cfg.wandb_enabled:
                log_dict = {}
                log_dict["loss"] = loss.item()
                log_dict.update(get_gradient_stats(encoder_decoder.parameters()))

                # Log control loss if needed
                if (train_iter + 1) % train_cfg.log_loss_control_every_n_iter == 0:
                    loss_control = get_loss_control(
                        train_cfg,
                        device,
                        encoder_decoder_ddp,
                        target_generated_tokens,
                        attention_mask,
                        tokenizer,
                        train_iter,
                    )
                    log_dict["loss_control"] = loss_control.item()

                wandb.log(log_dict)

            del target_acts, target_generated_tokens, attention_mask, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        # ------------------------------------------------------------------
        # Gracefully close the data-loader so that all background workers
        # (and their gopen pipes) terminate before Python leaves.
        # ------------------------------------------------------------------
        try:
            if "data_loader" in locals():
                # `iter(loader)` returns a DataLoader *iterator* whose private
                # method `_shutdown_workers()` stops the worker processes.
                if hasattr(data_loader, "_shutdown_workers"):
                    data_loader._shutdown_workers()

                # WebDataset itself may still hold open streams; close them too
                if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "close"):
                    data_loader.dataset.close()
        except Exception as e:
            print(f"Failed to close dataset cleanly: {e}")

        if train_cfg.save_encoder_path and rank == 0:
            save_encoder(encoder_decoder.module, train_cfg.save_encoder_path)
        
        dist.destroy_process_group()

        if train_cfg.wandb_enabled:
            wandb.finish()


