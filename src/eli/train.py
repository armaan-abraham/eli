import os
from typing import Dict

import torch
from tqdm import tqdm

import wandb
from eli.config import CPU, cfg, encoder_cfg
from eli.data import DataCollector
from eli.encoder import EncoderTrainer
from eli.utils import print_gpu_memory_usage
import gc

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

    wandb.log(
        {"loss_control": encoder_trainer.loss_control(data_collector, train_iter)}
    )

    for i in range(n_iter):
        log_dict = {}
        for key, val in metrics.items():
            log_dict[key] = val[i]
        wandb.log(log_dict)


def train():
    print("Starting training")
    print_gpu_memory_usage()
    data_collector = DataCollector(cfg)
    print("Data collector created")
    print_gpu_memory_usage()
    encoder_trainer = EncoderTrainer(cfg, encoder_cfg)
    print("Encoder trainer created")
    print_gpu_memory_usage()

    # Initialize wandb only if enabled
    if cfg.wandb_enabled:
        wandb.init(project=cfg.wandb_project)
        wandb.config.update(cfg)
        wandb.config.update(encoder_cfg)
    
    print("Wandb initialized")
    print_gpu_memory_usage()

    try:
        data_collector.prefetch_next_tokens()
        print("Next tokens prefetched")
        print_gpu_memory_usage()

        for train_iter in tqdm(range(cfg.num_train_iter), desc="Training"):
            collect_data(data_collector)
            print("Data collected")
            gc.collect()
            torch.cuda.empty_cache()
            print_gpu_memory_usage()

            encoder_trainer.move_models_to_device(cfg.device)
            print("Models moved to device")
            print_gpu_memory_usage()
            metrics = optimize_encoder(data_collector, encoder_trainer, train_iter)
            print("Encoder optimized")
            print_gpu_memory_usage()

            log_metrics(metrics, data_collector, encoder_trainer, train_iter)
            print("Metrics logged")
            print_gpu_memory_usage()
            encoder_trainer.move_models_to_device(CPU)
            print("Models moved to CPU")
            print_gpu_memory_usage()

    finally:
        data_collector.finish()
        
        if cfg.wandb_enabled:
            wandb.finish()

        print_gpu_memory_usage()

        if cfg.save_encoder_path:
            save_path = cfg.save_encoder_path

            save_path.parent.mkdir(parents=True, exist_ok=True)
            encoder_trainer.save_encoder(save_path)

if __name__ == "__main__":
    train()
