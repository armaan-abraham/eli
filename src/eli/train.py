import gc
import os
from typing import Dict

import torch
from tqdm import tqdm

import wandb
from eli.config import CPU, cfg, encoder_cfg
from eli.data import DataCollector
from eli.encoder import EncoderTrainer
from eli.utils import print_gpu_memory_usage

# I think there may be a bug in huggingface, just force
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def collect_data(data_collector: DataCollector):
    data_collector.collect_data()


def optimize_encoder(
    data_collector: DataCollector, encoder_trainer: EncoderTrainer, train_iter: int
):
    metrics = encoder_trainer.train(data_collector, train_iter)
    return metrics


def log_metrics(
    metrics: Dict,
    data_collector: DataCollector,
    encoder_trainer: EncoderTrainer,
    train_iter: int,
):
    if not cfg.wandb_enabled:
        return

    n_iter = len(metrics[list(metrics.keys())[0]])

    loss = metrics["loss"][0]
    loss_control = encoder_trainer.loss_control(data_collector, train_iter)

    wandb.log(
        {
            "loss_control": loss_control,
            "loss_normal": loss / loss_control,
        }
    )

    # Log per-batch metrics
    for i in range(n_iter):
        log_dict = {}
        for key, val in metrics.items():
            if isinstance(val, list):
                log_dict[key] = val[i]
        wandb.log(log_dict)

    # Log per-buffer metrics
    wandb.log({k: v for k, v in metrics.items() if not isinstance(v, list)})


def train():
    data_collector = DataCollector(cfg)
    encoder_trainer = EncoderTrainer(cfg, encoder_cfg)

    # Initialize wandb only if enabled
    if cfg.wandb_enabled:
        wandb.init(project=cfg.wandb_project)
        wandb.config.update(cfg)
        wandb.config.update(encoder_cfg)

    try:
        data_collector.prefetch_next_tokens()

        for train_iter in tqdm(range(cfg.num_train_iter), desc="Training"):
            collect_data(data_collector)

            gc.collect()
            torch.cuda.empty_cache()

            encoder_trainer.move_models_to_device(cfg.device)
            metrics = optimize_encoder(data_collector, encoder_trainer, train_iter)

            log_metrics(metrics, data_collector, encoder_trainer, train_iter)

            encoder_trainer.move_models_to_device(CPU)

            gc.collect()
            torch.cuda.empty_cache()

    finally:
        data_collector.finish()

        if cfg.wandb_enabled:
            wandb.finish()

        if cfg.save_encoder_path:
            save_path = cfg.save_encoder_path

            save_path.parent.mkdir(parents=True, exist_ok=True)
            encoder_trainer.save_encoder(save_path)


if __name__ == "__main__":
    train()
