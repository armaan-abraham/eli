import logging

from eli.datasets.config import ds_cfg
from eli.datasets.target import stream_target_data
from eli.datasets.tokens import stream_tokens
from eli.datasets.upload import create_and_upload_shards, upload_dataset_config
from eli.datasets.logging_utils import configure_logging


def main():
    dataset_name = f"{ds_cfg.target_model_name.replace('/', '-')}-{ds_cfg.site}-{ds_cfg.target_acts_layer_range[0]}-{ds_cfg.target_acts_layer_range[1]}-{ds_cfg.num_samples}"

    # Configure logging to both console and S3
    s3_handler = configure_logging(ds_cfg.s3_bucket, f"datasets/{dataset_name}")
    
    logging.info(f"Starting data collection for dataset: {dataset_name}")
    
    # Collect tokens
    token_stream = stream_tokens()

    try:
        # Upload dataset configuration
        upload_dataset_config(dataset_name)
        logging.info("Dataset configuration uploaded successfully")

        # Process tokens through target model
        target_stream = stream_target_data(token_stream)

        # Upload processed data
        create_and_upload_shards(target_stream, dataset_name)

    except Exception as e:
        logging.error(f"Error during data collection: {str(e)}", exc_info=True)
        raise

    finally:
        # Ensure any resources are properly closed
        if "target_stream" in locals() and hasattr(target_stream, "close"):
            target_stream.close()
        
        # Flush and close the S3 log handler
        logging.info("Finalizing logs...")
        s3_handler.flush()
        logging.getLogger().removeHandler(s3_handler)
        s3_handler.close()
