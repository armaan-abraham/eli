from eli.train.download import download_dataset
from transformers import AutoTokenizer




if __name__ == "__main__":
    loader = download_dataset(s3_bucket="eli-datasets", dataset_name="test")

    _, tokens = next(iter(loader))

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    
    for i in range(tokens.shape[0]):
        print(tokenizer.decode(tokens[i]))


