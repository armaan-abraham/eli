import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List

import aiohttp
import einops
import numpy as np
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer

from eli.datasets.config import DatasetConfig, ds_cfg

this_dir = Path(__file__).parent
cache_dir = this_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_tokenizer(ds_cfg: DatasetConfig = ds_cfg):
    # Handles custom schlepping and idiosyncracies for loading the currently specified tok
    tokenizer = AutoTokenizer.from_pretrained(ds_cfg.target_model_name)
    return tokenizer


def keep_single_column(dataset: Dataset, col_name: str) -> Dataset:
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name.

    Args:
        dataset: HuggingFace dataset to modify
        col_name: Name of the column to keep

    Returns:
        Dataset with only the specified column
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
) -> Dataset:
    """
    Tokenize and concatenate a dataset of text.

    This converts text to tokens, concatenates them (separated by EOS tokens) and reshapes
    them into a 2D array, dropping the last incomplete batch.

    Args:
        dataset: The dataset to tokenize (HuggingFace text dataset)
        tokenizer: The tokenizer with bos_token_id and eos_token_id
        streaming: Whether the dataset is being streamed
        max_length: Context window length
        column_name: Name of the text column in the dataset
        add_bos_token: Whether to add beginning of sequence token
        num_proc: Number of processes for parallel processing

    Returns:
        Tokenized dataset with a single column "tokens"
    """
    if tokenizer.pad_token is None:
        # Add padding token for tokenization (will be removed before using tokens in model)
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Define sequence length accounting for BOS token if needed
    seq_len = max_length - 1 if add_bos_token else max_length

    logging.info("Tokenize and concatenate called")

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        text = examples[column_name]
        assert isinstance(text, list), f"Expected list, got {type(text)}"
        assert isinstance(text[0], list), f"Expected list of lists, got {type(text[0])}"
        assert isinstance(text[0][0], str), (
            f"Expected list of lists of strings, got {type(text[0][0])}"
        )

        # Concatenate text with EOS tokens between entries
        full_text = tokenizer.eos_token.join(
            [tokenizer.eos_token.join(sub_text) for sub_text in text]
        )
        logging.info(f"Full text length: {len(full_text)}")

        # Handle empty text case
        if not full_text.strip():
            return {"tokens": np.array([], dtype=np.int64)}

        # Divide into chunks for parallel tokenization
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]

        # Tokenize chunks in parallel
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)

        assert num_tokens > seq_len, (
            f"Num tokens: {num_tokens} is less than seq_len: {seq_len}"
        )
        logging.info(f"Num tokens: {num_tokens}")

        # Create batches of tokens of length seq_len
        num_batches = num_tokens // seq_len
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )

        # Drop sequences that end with an EOS token (for generation)
        tokens = tokens[tokens[:, -1] != tokenizer.eos_token_id]

        # Add BOS token if required
        if add_bos_token:
            prefix = np.full((len(tokens), 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)

        return {"tokens": tokens}

    # Set up mapping parameters
    kwargs = {
        "batched": True,
        "remove_columns": [column_name],
    }
    if not streaming:
        kwargs["num_proc"] = num_proc

    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_function, **kwargs)
    return tokenized_dataset.with_format(type="torch")


def stream_tokens(ds_cfg: DatasetConfig = ds_cfg):
    """
    Stream tokenized chunks from a dataset for training.

    Args:
        ds_cfg: Configuration object

    Yields:
        Batches of tokenized text as tensors
    """
    if ds_cfg.use_fake_tokens:
        yield from stream_fake_tokens(ds_cfg)
        return

    CLIENT_TIMEOUT_SECONDS = 60 * 60 * 2
    storage_options = {
        "client_kwargs": {
            "timeout": aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_SECONDS),
        }
    }

    # Load and prepare dataset
    dataset_iter = load_dataset(
        ds_cfg.dataset_name,
        "en",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
        storage_options=storage_options,
    )

    dataset_iter = keep_single_column(dataset_iter, ds_cfg.dataset_column_name)
    dataset_iter = dataset_iter.batch(ds_cfg.dataset_batch_size_entries)
    dataset_iter = dataset_iter.shuffle(
        seed=ds_cfg.seed, buffer_size=ds_cfg.dataset_batch_size_entries
    )

    # Tokenize the dataset
    tokenizer = load_tokenizer()
    dataset_iter = tokenize_and_concatenate(
        dataset_iter,
        tokenizer,
        streaming=True,
        max_length=ds_cfg.target_ctx_len_toks,
        add_bos_token=True,
        column_name=ds_cfg.dataset_column_name,
    )

    dataset_iter = dataset_iter.batch(ds_cfg.dataset_entry_size_samples)

    for batch in dataset_iter:
        yield batch["tokens"].to(dtype=torch.int32, device="cpu")


def stream_fake_tokens(ds_cfg: DatasetConfig = ds_cfg) -> Iterator[torch.Tensor]:
    """
    Generate fake tokens for testing or benchmarking purposes.

    Args:
        ds_cfg: Configuration object

    Yields:
        Batches of random token IDs as tensors
    """
    # Set random seed for reproducibility
    torch.manual_seed(ds_cfg.seed)

    # Get vocabulary size from tokenizer or config
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)

    # Generate random tokens in the same shape as expected from real data
    while True:
        # Shape: [buffer_size_samples, target_ctx_len_toks]
        batch_shape = (ds_cfg.dataset_entry_size_samples, ds_cfg.target_ctx_len_toks)

        # Generate random token IDs within vocabulary range
        random_tokens = torch.randint(
            low=0, high=vocab_size, size=batch_shape, dtype=torch.int32, device="cpu"
        )

        yield random_tokens
