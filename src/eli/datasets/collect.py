from eli.datasets.target import stream_target_data
from eli.datasets.tokens import stream_tokens
from eli.datasets.upload import create_and_upload_shards, upload_dataset_config


def main(dataset_name: str):
    # Collect tokens
    token_stream = stream_tokens()

    try:
        # Process tokens through target model
        target_stream = stream_target_data(token_stream)

        # Upload processed data
        create_and_upload_shards(target_stream, dataset_name)
        
        # Upload dataset configuration
        upload_dataset_config(dataset_name)
    finally:
        # Ensure any resources are properly closed
        if "target_stream" in locals() and hasattr(target_stream, "close"):
            target_stream.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    args = parser.parse_args()
    main(args.dataset_name)
