from eli.datasets.config import ds_cfg
from eli.datasets.target import stream_target_data
from eli.datasets.tokens import stream_tokens
from eli.datasets.upload import create_and_upload_shards, upload_dataset_config


def main():
    dataset_name = f"{ds_cfg.target_model_name.replace('/', '-')}-{ds_cfg.site}-{ds_cfg.target_acts_layer_range[0]}-{ds_cfg.target_acts_layer_range[1]}"

    # Collect tokens
    token_stream = stream_tokens()

    try:
        # Upload dataset configuration
        upload_dataset_config(dataset_name)

        # Process tokens through target model
        target_stream = stream_target_data(token_stream)

        # Upload processed data
        create_and_upload_shards(target_stream, dataset_name)

    finally:
        # Ensure any resources are properly closed
        if "target_stream" in locals() and hasattr(target_stream, "close"):
            target_stream.close()
