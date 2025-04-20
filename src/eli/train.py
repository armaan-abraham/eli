import os
from typing import Dict

from tqdm import tqdm

import wandb
from eli.config import CPU, Config, EncoderConfig, cfg, encoder_cfg
from eli.context import DataCollectorEncoderContext
from eli.data import DataCollector
from eli.encoder import Encoder, EncoderTrainer

# I think there may be a bug in huggingface, just force
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def collect_data(data_collector: DataCollector):
    data_collector.collect_data()


def optimize_encoder(data_collector: DataCollector, encoder_trainer: EncoderTrainer):
    metrics = encoder_trainer.train(data_collector)
    return metrics


def log_metrics(
    metrics: Dict, data_collector: DataCollector, encoder_trainer: EncoderTrainer
):
    n_iter = len(metrics[list(metrics.keys())[0]])

    for i in range(n_iter):
        for key, val in metrics.items():
            wandb.log({key: val[i]})

    wandb.log({"loss_control": encoder_trainer.loss_control(data_collector)})


def train():
    data_encoder_context = DataCollectorEncoderContext(cfg)
    data_collector = DataCollector(data_encoder_context, cfg)
    encoder_trainer = EncoderTrainer(data_encoder_context, cfg, encoder_cfg)

    wandb.init(project="eli")
    wandb.config.update(cfg)
    wandb.config.update(encoder_cfg)

    try:
        data_collector.prefetch_next_tokens()
        for train_iter in tqdm(range(cfg.num_train_iter), desc="Training"):
            collect_data(data_collector)

            # Keep models on device until metrics are logged
            encoder_trainer.move_models_to_device(cfg.device)
            metrics = optimize_encoder(data_collector, encoder_trainer)

            log_metrics(metrics, data_collector, encoder_trainer)
            encoder_trainer.move_models_to_device(CPU)
    finally:
        data_collector.finish()
        wandb.finish()


if __name__ == "__main__":
    train()
