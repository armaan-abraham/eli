import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import asdict

import boto3
import numpy as np
import psutil
import torch
from tqdm import tqdm

from eli.datasets.config import ds_cfg


def create_and_upload_shards(
    tensor_batch_iterator,
    dataset_name,
):
    """
    Stream data from a tensor batch iterator, create WebDataset shards, and upload to S3.
    Each sample is added to the shard as a single entry.
    Processing stops after reaching the number of samples specified in the config.

    Parameters:
    - tensor_batch_iterator: Iterator yielding {"table_name": tensor} dictionaries
                            where each tensor has shape [batch_size, features]
    - dataset_name: S3 key prefix for uploads
    """
    # Create S3 client
    s3_client = boto3.client("s3")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Statistics tracking
    total_batches = 0
    total_samples = 0
    total_bytes_processed = 0

    # Define a shard naming pattern
    shard_pattern = os.path.join(temp_dir, "shard-%06d.tar")

    # Track uploaded shards
    uploaded_shards = []

    def upload_shard(fname):
        assert os.path.exists(fname), f"Shard {fname} does not exist"

        nonlocal uploaded_shards

        # Create index using widsindex command line tool
        index_path = fname + ".idx"
        subprocess.run(["widsindex", "create", fname, "-o", index_path], check=True)

        # Upload shard to S3
        shard_idx = len(uploaded_shards)
        s3_shard_key = f"datasets/{dataset_name}/{shard_idx:06d}.tar"
        s3_index_key = f"datasets/{dataset_name}/{shard_idx:06d}.tar.idx"

        shard_size_gb = os.path.getsize(fname) / 1024**3
        print(
            f"Uploading shard {shard_idx} ({shard_size_gb:.2f} GB) to s3://{ds_cfg.s3_bucket}/{s3_shard_key}"
        )

        s3_client.upload_file(fname, ds_cfg.s3_bucket, s3_shard_key)
        s3_client.upload_file(index_path, ds_cfg.s3_bucket, s3_index_key)

        # Clean up local files
        os.remove(fname)
        os.remove(index_path)

        uploaded_shards.append(fname)

        # Memory usage info
        memory_percent = psutil.virtual_memory().percent
        print(
            f"Memory usage: {memory_percent:.1f}%, Total samples: {total_samples}/{ds_cfg.num_samples}"
        )

    try:
        current_shard_idx = 0
        current_shard_path = shard_pattern % current_shard_idx
        current_shard_size = 0
        current_tar = tarfile.open(current_shard_path, "w")

        # Create a progress bar with target number of samples
        progress_bar = tqdm(total=ds_cfg.num_samples, desc="Processing samples")

        # Process batches
        for batch_idx, tensor_dict in enumerate(tensor_batch_iterator):
            # Get batch size from the first tensor
            table_names = list(tensor_dict.keys())
            assert table_names, "Empty batch"

            first_tensor = tensor_dict[table_names[0]]
            batch_size = first_tensor.shape[0]

            done = False

            # Process each sample in the batch
            for sample_idx in range(batch_size):
                # Check if we've reached the desired number of samples
                if total_samples >= ds_cfg.num_samples:
                    done = True
                    break

                # Create a unique key for each sample
                key = f"sample_{total_samples:08d}"
                sample_size_bytes = 0

                # Process each table tensor for this sample
                for table_name, tensor in tensor_dict.items():
                    # Extract the sample from the batch tensor
                    sample_tensor = tensor[
                        sample_idx : sample_idx + 1
                    ]  # Keep dimension

                    # Convert tensor to numpy and save to bytes
                    tensor_bytes = io.BytesIO()
                    np.save(tensor_bytes, sample_tensor.detach().cpu().numpy())
                    tensor_bytes.seek(0)

                    # Get the tensor data as bytes
                    tensor_bytes_data = tensor_bytes.read()

                    # Create a tarinfo for this file
                    info = tarfile.TarInfo(f"{key}/{table_name}.npy")
                    assert isinstance(
                        tensor_bytes_data, bytes
                    ), f"Data for {table_name} is not bytes"
                    file_data = io.BytesIO(tensor_bytes_data)
                    info.size = len(tensor_bytes_data)

                    file_data.seek(0)
                    current_tar.addfile(info, file_data)
                    current_shard_size += info.size
                    sample_size_bytes += info.size

                # Update tracking
                total_samples += 1
                total_bytes_processed += sample_size_bytes

                # Update progress bar
                progress_bar.update(1)

                # Check if we need to rotate to a new shard
                if current_shard_size >= ds_cfg.desired_shard_size_bytes:
                    # Close current shard
                    current_tar.close()

                    # Upload the completed shard
                    upload_shard(current_shard_path)

                    # Start a new shard
                    current_shard_idx += 1
                    current_shard_path = shard_pattern % current_shard_idx
                    current_shard_size = 0
                    current_tar = tarfile.open(current_shard_path, "w")

            # Update batch counter
            total_batches += 1

            if done:
                break

        # Close the progress bar
        progress_bar.close()

        # Close the final shard if it exists and has content
        if current_shard_size > 0:
            current_tar.close()
            upload_shard(current_shard_path)

        num_shards = len(uploaded_shards)
        print(
            f"Dataset creation complete: {num_shards} shards, {total_batches} batches, {total_samples}/{ds_cfg.num_samples} samples, {total_bytes_processed/1024/1024/1024:.2f} GB"
        )
        if num_shards > 0:
            print(
                f"Average shard size: {total_bytes_processed/num_shards/1024/1024:.2f} MB"
            )

        # Return statistics
        shard_stats = {
            "num_shards": len(uploaded_shards),
            "total_batches": total_batches,
            "total_samples": total_samples,
            "total_bytes": total_bytes_processed,
            "avg_shard_size_bytes": total_bytes_processed / len(uploaded_shards)
            if uploaded_shards
            else 0,
            "avg_sample_size_bytes": total_bytes_processed / total_samples
            if total_samples > 0
            else 0,
        }

        return shard_stats

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


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
