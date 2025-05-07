import io
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict

import boto3
import numpy as np
import psutil
import torch
from tqdm import tqdm
from webdataset import ShardWriter

from eli.datasets.config import ds_cfg


def create_and_upload_shards(
    tensor_batch_iterator,
    dataset_name,
):
    writer = ShardWriter(
        f"pipe:cat - | aws s3 cp - s3://{ds_cfg.s3_bucket}/datasets/{dataset_name}/%08d.tar",
        maxsize=ds_cfg.max_shard_size_bytes,
    )

    progress_bar = tqdm(total=ds_cfg.num_samples, desc="Processing samples")

    total_samples = 0
    done = False

    for tensor_dict in tensor_batch_iterator:
        # Get batch size from the first tensor
        table_names = list(tensor_dict.keys())
        assert table_names, "Empty batch"

        first_tensor = tensor_dict[table_names[0]]
        batch_size = first_tensor.shape[0]

        # Process each sample in the batch
        for sample_idx in range(batch_size):
            sample_dict = {
                "__key__": f"sample_{total_samples:08d}",
            }

            # Process each table tensor for this sample
            for table_name, tensor in tensor_dict.items():
                # Extract the sample from the batch tensor
                sample_tensor = tensor[sample_idx]
                sample_dict[f"{table_name}.pth"] = sample_tensor

            writer.write(sample_dict)

            # Update tracking
            total_samples += 1

            # Update progress bar
            progress_bar.update(1)

            # Check if we've reached the desired number of samples
            if total_samples >= ds_cfg.num_samples:
                done = True
                break

        if done:
            break

    writer.close()


def upload_dataset_config(dataset_name):
    """
    Upload the dataset configuration to S3.

    Parameters:
    - dataset_name: S3 key prefix for upload

    Returns:
    - Dictionary with information about the upload
    """
    # Create S3 client
    s3_client = boto3.client("s3")

    # Convert dataset config to dictionary
    config_dict = asdict(ds_cfg)

    # No need to handle torch.device and torch.dtype anymore
    # They're already stored as strings in _device_str and _dtype_str

    # Convert to JSON
    config_json = json.dumps(config_dict, indent=2)

    # Upload to S3
    s3_key = f"datasets/{dataset_name}/config.json"
    s3_client.put_object(
        Bucket=ds_cfg.s3_bucket,
        Key=s3_key,
        Body=config_json,
        ContentType="application/json",
    )

    print(f"Uploaded dataset configuration to s3://{ds_cfg.s3_bucket}/{s3_key}")

    return {"config_size_bytes": len(config_json), "config_s3_key": s3_key}
