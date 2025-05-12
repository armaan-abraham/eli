import io
import os
from typing import Any, Dict, Iterator

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader

from eli.train.config import train_cfg


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
    
    s3 = boto3.client('s3')
    prefix = f"datasets/{dataset_name}/"
    
    # List objects with the given prefix
    response = s3.list_objects_v2(
        Bucket=s3_bucket,
        Prefix=prefix
    )
    
    # Count only the .tar files
    shard_count = 0
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.tar'):
                shard_count += 1
                
    return shard_count


def download_dataset(
    s3_bucket: str, dataset_name: str
) -> Iterator[Dict[str, torch.Tensor]]:

    # Get the number of shards
    shard_count = get_shard_count(s3_bucket, dataset_name)
    
    # Create a range string based on the actual count (e.g., "00000000..00000015")
    shard_range = f"{{00000000..{shard_count-1:08d}}}"
    
    # Create S3 URL pattern with the dynamic range
    url = f"pipe: aws s3 cp s3://{s3_bucket}/datasets/{dataset_name}/{shard_range}.tar -"

    # Create WebDataset pipeline
    dataset = (
        wds.WebDataset(url, shardshuffle=False)
        .decode()
        .to_tuple("target_acts.pth", "target_generated_tokens.pth")
        .batched(train_cfg.dataset_loader_batch_size)
    )

    # TODO: fix batch size issue with increasing workers
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=min(shard_count, 4),
    )

    return loader
