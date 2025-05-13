import io
import os
from typing import Any, Dict, Iterator

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader

from eli.datasets.config import DatasetConfig
from eli.train.config import TrainConfig, train_cfg


def get_shard_count(s3_bucket: str, dataset_name: str) -> int:
    """
    Dynamically determine the number of shards available in the S3 bucket.

    Args:
        s3_bucket: Name of the S3 bucket
        dataset_name: Name of the dataset

    Returns:
        int: Number of available shards
    """
    import boto3

    s3 = boto3.client("s3")
    prefix = f"datasets/{dataset_name}/"

    # List objects with the given prefix
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)

    # Count only the .tar files
    shard_count = 0
    if "Contents" in response:
        for obj in response["Contents"]:
            if obj["Key"].endswith(".tar"):
                shard_count += 1

    return shard_count


def download_dataset(
    dataset_cfg: DatasetConfig, train_cfg: TrainConfig = train_cfg
) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Inspired by
    https://github.com/webdataset/webdataset/blob/main/examples/train-resnet50-multiray-wds.ipynb
    """

    # Get the number of shards
    shard_count = get_shard_count(train_cfg.s3_bucket, train_cfg.dataset_name)

    # Create a range string based on the actual count (e.g., "00000000..00000015")
    shard_range = f"{{00000000..{shard_count - 1:08d}}}"

    # Create S3 URL pattern with the dynamic range
    url = f"pipe: aws s3 cp s3://{train_cfg.s3_bucket}/datasets/{train_cfg.dataset_name}/{shard_range}.tar -"

    print(f"URL: {url}")

    # Create WebDataset pipeline
    dataset = (
        # Shard shuffle false because of multi node
        wds.WebDataset(
            url, resampled=True, nodesplitter=wds.split_by_node, shardshuffle=False
        )
        .shuffle(train_cfg.dataset_loader_shuffle_buffer_size_wds_entries)
        .decode()
        .to_tuple("target_acts.pth", "target_generated_tokens.pth")
        .unbatched()  # Unbatch because entries contain multiple samples
        .batched(train_cfg.dataset_loader_batch_size_samples)
    )

    loader = (
        wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=min(shard_count, 4),
        )
        # Unbatch, shuffle, batch again to mix samples from different workers
        .unbatched()
        # Shuffle the same number of samples as first shuffle
        .shuffle(
            train_cfg.dataset_loader_shuffle_buffer_size_wds_entries
            * dataset_cfg.dataset_entry_size_samples
        )
        .batched(train_cfg.dataset_loader_batch_size_samples)
    )

    return loader
