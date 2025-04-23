import os
from typing import Dict

from tqdm import tqdm

import wandb
from eli.config import CPU, cfg, encoder_cfg
from eli.data import DataCollector
from eli.encoder import EncoderTrainer

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
    data_collector = DataCollector(cfg)
    encoder_trainer = EncoderTrainer(cfg, encoder_cfg)

    wandb.init(project="eli")
    wandb.config.update(cfg)
    wandb.config.update(encoder_cfg)

    try:
        data_collector.prefetch_next_tokens()
        for train_iter in tqdm(range(cfg.num_train_iter), desc="Training"):
            collect_data(data_collector)

            # Keep models on device until metrics are logged
            encoder_trainer.move_models_to_device(cfg.device)
            metrics = optimize_encoder(data_collector, encoder_trainer, train_iter)

            log_metrics(metrics, data_collector, encoder_trainer, train_iter)
            encoder_trainer.move_models_to_device(CPU)
    finally:
        data_collector.finish()
        wandb.finish()


if __name__ == "__main__":
    train()
