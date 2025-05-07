from eli.train.download import download_dataset

if __name__ == "__main__":
    loader = download_dataset(s3_bucket="eli-datasets", dataset_name="test")

    batch = next(iter(loader))

    print(batch[0].shape, batch[1].shape)
