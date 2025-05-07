import io
import os
from typing import Any, Dict, Iterator

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader

from eli.train.config import train_cfg


def download_dataset(
    s3_bucket: str, dataset_name: str
) -> Iterator[Dict[str, torch.Tensor]]:
    # Create S3 URL pattern for the shards
    url = f"pipe: aws s3 cp s3://{s3_bucket}/datasets/{dataset_name}/{{00000000..00000015}}.tar -"

    # Create WebDataset pipeline
    dataset = (
        wds.WebDataset(url, shardshuffle=False)
        .decode()
        .to_tuple("target_acts.pth", "target_generated_tokens.pth")
        .batched(train_cfg.dataset_loader_batch_size)
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
    )

    return loader
