from typing import Dict

import wandb
from tqdm import tqdm

from eli.config import CPU, Config, EncoderConfig, cfg, encoder_cfg
from eli.data import DataCollector
from eli.encoder import Encoder, EncoderTrainer


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
    data_collector = DataCollector(cfg)
    encoder_trainer = EncoderTrainer(cfg, encoder_cfg)

    wandb.init(project="eli")
    wandb.config.update(cfg)
    wandb.config.update(encoder_cfg)

    try:
        for train_iter in tqdm(range(cfg.num_train_iter), desc="Training"):
            collect_data(data_collector)

            # Keep models on device until metrics are logged
            encoder_trainer.move_models_to_device(cfg.device)
            metrics = optimize_encoder(data_collector, encoder_trainer)

            log_metrics(metrics, data_collector, encoder_trainer)
            encoder_trainer.move_models_to_device(CPU)
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
