import os
from typing import Dict

import torch
from tqdm import tqdm

import wandb
from eli.config import CPU, cfg, encoder_cfg
from eli.data import DataCollector
from eli.encoder import EncoderTrainer
import torch.profiler

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
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/train'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        data_collector = DataCollector(cfg)
        encoder_trainer = EncoderTrainer(cfg, encoder_cfg)

        wandb.init(project="eli")
        wandb.config.update(cfg)
        wandb.config.update(encoder_cfg)

        try:
            data_collector.prefetch_next_tokens()

            torch.cuda.memory._record_memory_history(
                max_entries=int(1e5)
            )

            for train_iter in tqdm(range(cfg.num_train_iter), desc="Training"):
                collect_data(data_collector)

                encoder_trainer.move_models_to_device(cfg.device)
                metrics = optimize_encoder(data_collector, encoder_trainer, train_iter)

                log_metrics(metrics, data_collector, encoder_trainer, train_iter)
                encoder_trainer.move_models_to_device(CPU)


            torch.cuda.memory._dump_snapshot(f"train.pickle")

        finally:

            data_collector.finish()
            wandb.finish()

            if cfg.save_encoder_path:
                save_path = cfg.save_encoder_path

                save_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Saving encoder to {save_path}")
                encoder_trainer.save_encoder(save_path)

    if prof:
        if torch.cuda.is_available():
            pass
            # num_devices = torch.cuda.device_count()
            # print(f"Found {num_devices} CUDA device(s). Exporting memory timelines...")
            # # for i in range(num_devices):
            # device_str = f"cuda:{0}"
            # try:
            #     timeline_dir = "./memory_timelines"
            #     os.makedirs(timeline_dir, exist_ok=True)
            #     timeline_path = os.path.join(timeline_dir, f"memory_timeline_{device_str.replace(':', '')}.html")
            #     print(f"  Exporting memory timeline for {device_str} to {timeline_path}")
            #     prof.export_memory_timeline(timeline_path, device=device_str)
            #     print(f"  Exported memory timeline for {device_str} to {timeline_path}")
            # except Exception as e:
            #     print(f"  Failed to export memory timeline for {device_str}: {e}")
        else:
            print("Memory timeline export skipped: No CUDA devices available.")

if __name__ == "__main__":
    train()
