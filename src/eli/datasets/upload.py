import io
import os
import shutil
import tempfile

import boto3
import numpy as np
import psutil
import torch
import webdataset as wds
from tqdm import tqdm
from webdataset import ShardWriter

from eli.datasets.config import ds_cfg


def create_and_upload_shards(
    tensor_batch_iterator,
    dataset_name,
    target_shard_size_bytes=1024 * 1024 * 1024,  # 1GB
    temp_dir=None,
    verbose=True,
):
    """
    Stream data from a tensor batch iterator, create WebDataset shards, and upload to S3.
    Each batch is added to the shard as a single entry.

    Parameters:
    - tensor_batch_iterator: Iterator yielding {"table_name": tensor} dictionaries
                            where each tensor has shape [batch_size, features]
    - dataset_name: S3 key prefix for uploads
    - target_shard_size_bytes: Target size for each shard
    - temp_dir: Directory for temporary files
    - verbose: Whether to print progress information
    """
    # Create S3 client
    s3_client = boto3.client("s3")

    # Create temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)

    # Statistics tracking
    total_batches = 0
    total_samples = 0
    total_bytes_processed = 0

    # Define a shard naming pattern
    shard_pattern = os.path.join(temp_dir, "shard-%06d.tar")

    # Create a callback for when shards are completed
    uploaded_shards = []

    def upload_callback(fname):
        nonlocal uploaded_shards

        if not os.path.exists(fname):
            return

        # Create index
        index_path = fname + ".idx"
        wds.make_tar_index(fname, index_path)

        # Upload shard to S3
        shard_idx = len(uploaded_shards)
        s3_shard_key = f"datasets/{dataset_name}/{shard_idx:06d}.tar"
        s3_index_key = f"datasets/{dataset_name}/{shard_idx:06d}.tar.idx"

        if verbose:
            shard_size_mb = os.path.getsize(fname) / (1024 * 1024)
            print(
                f"Uploading shard {shard_idx} ({shard_size_mb:.2f} MB) to s3://{ds_cfg.s3_bucket}/{s3_shard_key}"
            )

        s3_client.upload_file(fname, ds_cfg.s3_bucket, s3_shard_key)

        if os.path.exists(index_path):
            s3_client.upload_file(index_path, ds_cfg.s3_bucket, s3_index_key)

        # Clean up local files
        os.remove(fname)
        if os.path.exists(index_path):
            os.remove(index_path)

        uploaded_shards.append(fname)

        # Memory usage info
        if verbose:
            memory_percent = psutil.virtual_memory().percent
            print(
                f"Memory usage: {memory_percent:.1f}%, Total samples: {total_samples}"
            )

    try:
        # Create ShardWriter with target size and upload callback
        with ShardWriter(
            shard_pattern,
            maxsize=target_shard_size_bytes,
            maxcount=None,  # No limit on samples per shard
            keep=False,  # Don't keep files locally
            postprocess=upload_callback,
        ) as writer:
            # Process batches
            for batch_idx, tensor_dict in enumerate(
                tqdm(tensor_batch_iterator) if verbose else tensor_batch_iterator
            ):
                # Get batch size from the first tensor
                table_names = list(tensor_dict.keys())
                if not table_names:
                    continue  # Skip empty batches

                first_tensor = tensor_dict[table_names[0]]
                batch_size = first_tensor.shape[0]

                # Create a batch entry - store the entire batch as one entry
                batch_data = {"__key__": f"batch_{total_batches:08d}"}

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

                # Write batch to current shard
                writer.write(batch_data)

                # Update tracking
                total_batches += 1
                total_samples += batch_size
                total_bytes_processed += batch_size_bytes

        if verbose:
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
