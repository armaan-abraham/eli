from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_samples: int = int(2e7)

    seed: int = 42

    dataset_loader_batch_size: int = 200

    webdataset_shardshuffle: int = 2


train_cfg = TrainConfig()
