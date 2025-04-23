import logging
import os
import threading
from pathlib import Path
from queue import Queue
from typing import Dict, List

import aiohttp
import einops
import numpy as np
import torch
import transformer_lens
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.config import CPU, Config, cfg
from eli.utils import log_gpu_memory_usage

this_dir = Path(__file__).parent
cache_dir = this_dir / "cache"

for dir_path in [cache_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)


def keep_single_column(dataset: Dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


# This is a modified version of the tokenize_and_concatenate function from transformer_lens.utils
def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
) -> Dataset:
    """Helper function to tokenize and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"
    """
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    logging.info(f"Tokenize and concatenate called")

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        text = examples[column_name]
        assert isinstance(text, list), f"Expected list, got {type(text)}"
        assert isinstance(text[0], list), f"Expected list of lists, got {type(text[0])}"
        assert isinstance(
            text[0][0], str
        ), f"Expected list of lists of strings, got {type(text[0][0])}"
        # Concatenate it all into an enormous string, separated by eos_tokens
        # This double loop looks incorrect, but we are actually getting a list
        # of lists of strings from text, so this is required and correct
        full_text = tokenizer.eos_token.join(
            [tokenizer.eos_token.join(sub_text) for sub_text in text]
        )
        logging.info(f"Full text length: {len(full_text)}")

        # Handle the case when full_text is empty
        if not full_text.strip():
            return {"tokens": np.array([], dtype=np.int64)}

        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        assert (
            num_tokens > seq_len
        ), f"Num tokens: {num_tokens} is less than seq_len: {seq_len}"
        logging.info(f"Num tokens: {num_tokens}")

        # Handle cases where num_tokens is less than seq_len
        # if num_tokens < seq_len:
        #     num_batches = 1
        #     # Pad tokens if necessary
        #     tokens = tokens[:seq_len]
        #     if len(tokens) < seq_len:
        #         padding_length = seq_len - len(tokens)
        #         padding = np.full(padding_length, tokenizer.pad_token_id)
        #         tokens = np.concatenate([tokens, padding], axis=0)
        # else:

        num_batches = num_tokens // seq_len

        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]

        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )

        # Drop sequences that end with an EOS token because we are using these
        # for generation.
        tokens = tokens[tokens[:, -1] != tokenizer.eos_token_id]

        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    kwargs = {
        "batched": True,
        "remove_columns": [column_name],
    }
    if not streaming:
        kwargs["num_proc"] = num_proc

    tokenized_dataset = dataset.map(
        tokenize_function,
        **kwargs,
    )
    return tokenized_dataset.with_format(type="torch")


def stream_training_chunks(
    cfg: Config = cfg,
):
    CLIENT_TIMEOUT_SECONDS = 60 * 60 * 2
    storage_options = {
        "client_kwargs": {
            "timeout": aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_SECONDS),
        }
    }
    dataset_iter = load_dataset(
        cfg.dataset_name,
        "en",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
        storage_options=storage_options,
    )

    dataset_iter = keep_single_column(dataset_iter, cfg.dataset_column_name)
    dataset_iter = dataset_iter.batch(cfg.dataset_batch_size_entries)

    dataset_iter = dataset_iter.shuffle(
        seed=cfg.seed, buffer_size=cfg.dataset_batch_size_entries
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name)

    dataset_iter = tokenize_and_concatenate(
        dataset_iter,
        tokenizer,
        streaming=True,
        max_length=cfg.target_ctx_len_toks,
        add_bos_token=False,
        column_name=cfg.dataset_column_name,
    )

    dataset_iter = dataset_iter.batch(cfg.buffer_size_samples)

    for batch in dataset_iter:
        yield batch["tokens"].to(dtype=torch.int32, device="cpu")


class DataCollector:
    def __init__(self, cfg: Config = cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name)
        self.token_stream = stream_training_chunks(cfg)

        # Add a queue to hold prefetched tokens
        self.token_queue = Queue(maxsize=1)
        self.prefetch_thread = None

        # Define buffers - using float32 by default for storage
        self.target_generated_tokens = torch.zeros(
            (self.cfg.buffer_size_samples, self.cfg.target_generation_len_toks),
            dtype=torch.int32,
            device=CPU,
        )
        self.target_logits = torch.zeros(
            (
                self.cfg.buffer_size_samples,
                self.cfg.decoder_pred_len_toks,
                self.cfg.vocab_size,
            ),
            dtype=torch.float32,  # Store in float32
            device=CPU,
        )
        self.target_acts = torch.zeros(
            (self.cfg.buffer_size_samples, self.cfg.target_model_act_dim),
            dtype=torch.float32,  # Store in float32
            device=CPU,
        )

        # Load models in float32
        self.target_model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model_name
        ).to(CPU)
        self.target_model_act_collection = (
            transformer_lens.HookedTransformer.from_pretrained(
                cfg.target_model_name
            ).to(CPU)
        )

    def prefetch_next_tokens(self):
        """Start a new thread to fetch the next batch of tokens"""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            # Wait for any existing prefetch to complete
            self.prefetch_thread.join()

        def fetch_worker():
            try:
                tokens = next(self.token_stream)
                self.token_queue.put(tokens)
            except StopIteration:
                # Handle end of dataset if needed
                self.token_queue.put(None)

        self.prefetch_thread = threading.Thread(target=fetch_worker)
        self.prefetch_thread.daemon = True  # Thread will exit when main program exits
        self.prefetch_thread.start()

    def move_models_to_device(self, device: torch.device):
        self.target_model.to(device)
        self.target_model_act_collection.to(device)

    @log_gpu_memory_usage
    def collect_data(self):
        # Get the prefetched tokens
        toks_init = self.token_queue.get()

        # Start prefetching the next batch
        self.prefetch_next_tokens()

        assert toks_init is not None, "Ran out of token data"
        assert toks_init.shape == (
            self.cfg.buffer_size_samples,
            self.cfg.target_ctx_len_toks,
        )

        num_batches = (
            self.cfg.buffer_size_samples // self.cfg.target_model_batch_size_samples
        )

        self.move_models_to_device(self.cfg.device)

        for i in range(num_batches):
            start_idx = i * self.cfg.target_model_batch_size_samples
            end_idx = start_idx + self.cfg.target_model_batch_size_samples

            # Get current batch
            batch_toks = toks_init[start_idx:end_idx].to(self.cfg.device)

            # Use autocast for all model operations
            with torch.autocast(device_type=self.cfg.device.type, dtype=self.cfg.dtype):
                # Collect acts of target model
                _, cache = self.target_model_act_collection.run_with_cache(
                    batch_toks,
                    stop_at_layer=self.cfg.layer + 1,
                    names_filter=self.cfg.act_name,
                    return_cache_object=True,
                )
                self.target_acts[start_idx:end_idx] = cache.cache_dict[
                    self.cfg.act_name
                ][:, -1, :]

                # Generate tokens with target model
                length_toks = (
                    self.cfg.target_ctx_len_toks + self.cfg.target_generation_len_toks
                )

                # Create an attention mask of all 1s (all tokens are valid content)
                attention_mask = torch.ones_like(batch_toks, dtype=torch.int32)

                batch_toks_with_gen = self.target_model.generate(
                    batch_toks,
                    attention_mask=attention_mask,
                    max_length=length_toks,
                    min_length=length_toks,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                self.target_generated_tokens[start_idx:end_idx] = batch_toks_with_gen[
                    :, -self.cfg.target_generation_len_toks :
                ]

                # Collect logits of target model
                with torch.no_grad():
                    self.target_logits[start_idx:end_idx] = self.target_model(
                        batch_toks_with_gen
                    ).logits[:, -self.cfg.decoder_pred_len_toks :, :]

        self.move_models_to_device(CPU)

    @property
    def data(self):
        return {
            "target_generated_tokens": self.target_generated_tokens,
            "target_logits": self.target_logits,
            "target_acts": self.target_acts,
        }

    def finish(self):
        """Clean up resources and join any pending threads"""
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join()
