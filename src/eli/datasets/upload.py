import json
import logging
from dataclasses import asdict

import boto3
from tqdm import tqdm
from webdataset import ShardWriter

from eli.datasets.config import ds_cfg
from eli.datasets.logging_utils import log_progress


def create_and_upload_shards(
    tensor_batch_iterator,
    dataset_name,
):
    writer = ShardWriter(
        f"pipe:cat - | aws s3 cp - s3://{ds_cfg.s3_bucket}/datasets/{dataset_name}/%08d.tar",
        maxsize=ds_cfg.max_shard_size_bytes,
    )
    progress_bar = tqdm(
        total=ds_cfg.num_samples, desc="Processing samples", unit="samples"
    )

    total_samples = 0
    batch_idx = 0

    total_bytes = 0
    logging.info(f"Starting to create and upload shards for {dataset_name}")

    for tensor_dict in tensor_batch_iterator:
        # Get batch size from the first tensor
        table_names = list(tensor_dict.keys())
        assert table_names, "Empty batch"

        first_tensor = tensor_dict[table_names[0]]
        batch_size = first_tensor.shape[0]

        for table_name, tensor in tensor_dict.items():
            assert (
                tensor.shape[0] == batch_size
            ), f"Batch size mismatch for table {table_name}"

        # Write the entire batch at once
        sample_dict = {
            "__key__": f"sample_{batch_idx:08d}",
        }

        # Add each tensor as a batch
        for table_name, tensor in tensor_dict.items():
            tensor_size_bytes = tensor.element_size() * tensor.numel()
            total_bytes += tensor_size_bytes
            sample_dict[f"{table_name}.npy"] = tensor.cpu().numpy()

        writer.write(sample_dict)

        # Update tracking
        total_samples += batch_size
        batch_idx += 1

        # Update progress bar with bytes information
        progress_bar.set_postfix(processed_bytes=f"{total_bytes / 1024**2:.2f} MB")
        progress_bar.update(batch_size)

        # Log progress to file (tqdm updates console)
        log_progress(total_samples, ds_cfg.num_samples)
        logging.info(f"Total data size: {total_bytes / 1024**2:.2f} MB")

        # Check if we've reached the desired number of samples
        if total_samples >= ds_cfg.num_samples:
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

    return {"config_size_bytes": len(config_json), "config_s3_key": s3_key}
