import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import time

import boto3
import numpy as np
import psutil
import torch
import webdataset as wds
from tqdm import tqdm

from eli.datasets.config import ds_cfg


def create_and_upload_shards(
    tensor_batch_iterator,
    dataset_name,
):
    """
    Stream data from a tensor batch iterator, create WebDataset shards, and upload to S3.
    Each batch is added to the shard as a single entry.

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
        print(f"Memory usage: {memory_percent:.1f}%, Total samples: {total_samples}")

    try:
        current_shard_idx = 0
        current_shard_path = shard_pattern % current_shard_idx
        current_shard_size = 0
        current_tar = tarfile.open(current_shard_path, "w")

        # Process batches
        for batch_idx, tensor_dict in enumerate(tqdm(tensor_batch_iterator)):
            # Get batch size from the first tensor
            table_names = list(tensor_dict.keys())
            assert table_names, "Empty batch"

            first_tensor = tensor_dict[table_names[0]]
            batch_size = first_tensor.shape[0]

            # Create a batch entry - store the entire batch as one entry
            key = f"batch_{total_batches:08d}"
            batch_data = {"__key__": key}

            batch_size_bytes = 0

            # Process each table tensor in the batch
            for table_name, tensor in tensor_dict.items():
                # Convert tensor to numpy and save to bytes
                tensor_bytes = io.BytesIO()
                np.save(tensor_bytes, tensor.detach().cpu().numpy())
                tensor_bytes.seek(0)

                # Add to batch data with appropriate extension
                tensor_bytes_data = tensor_bytes.read()
                batch_data[f"{table_name}.npy"] = tensor_bytes_data

                # Track size
                batch_size_bytes += len(tensor_bytes_data)

            # Write batch to current shard using TarFile
            for name, data in batch_data.items():
                if name == "__key__":
                    continue

                # Create a tarinfo for this file
                info = tarfile.TarInfo(f"{key}/{name}")
                assert isinstance(data, bytes), f"Data for {name} is not bytes"
                file_data = io.BytesIO(data)
                info.size = len(data)

                file_data.seek(0)
                current_tar.addfile(info, file_data)
                current_shard_size += info.size

            # Update tracking
            total_batches += 1
            total_samples += batch_size
            total_bytes_processed += batch_size_bytes

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

        # Close the final shard if it exists and has content
        if current_shard_size > 0:
            current_tar.close()
            upload_shard(current_shard_path)

        num_shards = len(uploaded_shards)
        print(
            f"Dataset creation complete: {num_shards} shards, {total_batches} batches, {total_samples} samples, {total_bytes_processed/1024/1024/1024:.2f} GB"
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
            "avg_batch_size_bytes": total_bytes_processed / total_batches
            if total_batches > 0
            else 0,
        }

        return shard_stats

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)
