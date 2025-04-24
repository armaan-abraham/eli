import logging
import os
import threading
import traceback
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import einops
import numpy as np
import torch
import torch.multiprocessing as mp
import transformer_lens
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.config import CPU, Config, cfg
from eli.utils import print_gpu_memory_usage

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


def worker_process(
    proc_idx: int,
    device: torch.device,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    cfg: Config,
    input_tokens: torch.Tensor,
    target_acts: torch.Tensor,
    target_generated_tokens: torch.Tensor,
    target_logits: torch.Tensor,
) -> None:
    """Worker process that loads models and processes batches"""
    try:
        # Load models in the worker process (each process gets its own copy)
        tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name)

        target_model = AutoModelForCausalLM.from_pretrained(cfg.target_model_name).to(
            CPU
        )
        target_model_act_collection = (
            transformer_lens.HookedTransformer.from_pretrained(
                cfg.target_model_name
            ).to(CPU)
        )

        logging.info(f"Worker {proc_idx} loaded models on {device}")

        while True:
            # Get chunk range from the queue
            chunk_range = task_queue.get()
            if chunk_range is None:  # Sentinel value to stop
                break

            target_model.to(device)
            target_model_act_collection.to(device)

            torch.cuda.reset_peak_memory_stats(device)

            chunk_start, chunk_end = chunk_range
            logging.info(
                f"Worker {proc_idx} processing chunk {chunk_start}:{chunk_end}"
            )
            
            # Process the chunk in batches
            batch_size = cfg.target_model_batch_size_samples
            for batch_start in range(chunk_start, chunk_end, batch_size):
                batch_end = min(batch_start + batch_size, chunk_end)
                logging.info(
                    f"Worker {proc_idx} processing batch {batch_start}:{batch_end}"
                )
                
                # Process the batch
                batch_toks = input_tokens[batch_start:batch_end].to(device)

                # Use autocast for model operations
                with torch.autocast(device_type=device.type, dtype=cfg.dtype):
                    # Collect activations
                    _, cache = target_model_act_collection.run_with_cache(
                        batch_toks,
                        stop_at_layer=cfg.layer + 1,
                        names_filter=cfg.act_name,
                        return_cache_object=True,
                    )

                    # Get activations and move to shared memory
                    acts = cache.cache_dict[cfg.act_name][:, -1, :]
                    target_acts[batch_start:batch_end] = acts.cpu()

                    # Generate tokens
                    length_toks = cfg.target_ctx_len_toks + cfg.target_generation_len_toks

                    # Create attention mask
                    attention_mask = torch.ones_like(batch_toks, dtype=torch.int32)

                    # Generate tokens
                    batch_toks_with_gen = target_model.generate(
                        batch_toks,
                        attention_mask=attention_mask,
                        max_length=length_toks,
                        min_length=length_toks,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    # Store generated tokens in shared memory
                    generated_tokens = batch_toks_with_gen[
                        :, -cfg.target_generation_len_toks :
                    ]
                    target_generated_tokens[batch_start:batch_end] = generated_tokens.cpu()

                    # Collect logits
                    with torch.no_grad():
                        logits = target_model(batch_toks_with_gen).logits[
                            :, -cfg.decoder_pred_len_toks :, :
                        ]
                        target_logits[batch_start:batch_end] = logits.cpu()
                    
                    del batch_toks_with_gen, logits, acts, cache

            peak_mem = torch.cuda.max_memory_allocated(device)
            logging.info(
                f"Worker {proc_idx} peak memory: {round(peak_mem / 1024**3, 1)} GB"
            )

            # Move to CPU before returning so we don't get an OOM error if we do
            # something on the GPU immediately after
            target_model.to(CPU)
            target_model_act_collection.to(CPU)

            # Explicitly clear cached memory on the GPU after processing a chunk
            # and moving models to CPU.
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Notify completion of the entire chunk
            result_queue.put(("success", chunk_start, chunk_end))

        logging.info(f"Worker {proc_idx} finished")

        # Clear cache one last time when worker exits normally
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        # Send the error back to the main process using the result queue
        error_str = f"Error in worker {proc_idx}: {str(e)}\n{traceback.format_exc()}"
        result_queue.put(("error", error_str))
        logging.error(error_str)
        # Clear cache on error exit as well
        if device.type == 'cuda':
             torch.cuda.empty_cache()


class DataCollector:
    def __init__(self, cfg: Config = cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name)
        
        # Only create the token stream if not using fake tokens
        if not self.cfg.use_fake_tokens:
            self.token_stream = stream_training_chunks(cfg)
            # Add a queue to hold prefetched tokens
            self.token_queue = Queue(maxsize=1)
            self.prefetch_thread = None
        else:
            logging.info("Using fake tokens for testing")
            self.token_stream = None
            self.token_queue = None
            self.prefetch_thread = None

        # Setup multiprocessing
        mp.set_start_method("spawn", force=True)

        # Detect available GPUs
        self.num_gpus = torch.cuda.device_count()
        # Use just one process if CPU-only (multiple CPU processes can cause memory issues)
        self.num_processes = self.num_gpus if self.num_gpus > 0 else 1
        self.devices = (
            [torch.device(f"cuda:{i}") for i in range(self.num_gpus)]
            if self.num_gpus > 0
            else [torch.device("cpu")]
        )

        logging.info(f"Using {self.num_processes} processes on devices: {self.devices}")

        # Create shared memory tensors
        self.setup_shared_tensors()

        # Initialize multiprocessing components
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = []

        # Start prefetching tokens
        self.prefetch_next_tokens()

        self.start_worker_processes()

    def setup_shared_tensors(self):
        """Create shared memory tensors for input and output data"""
        # Define shared memory tensors with proper dtype for storage
        self.target_generated_tokens = torch.zeros(
            (self.cfg.buffer_size_samples, self.cfg.target_generation_len_toks),
            dtype=torch.int32,
            device=CPU,
        ).share_memory_()

        self.target_logits = torch.zeros(
            (
                self.cfg.buffer_size_samples,
                self.cfg.decoder_pred_len_toks,
                self.cfg.vocab_size,
            ),
            dtype=torch.float32,
            device=CPU,
        ).share_memory_()

        self.target_acts = torch.zeros(
            (self.cfg.buffer_size_samples, self.cfg.target_model_act_dim),
            dtype=torch.float32,
            device=CPU,
        ).share_memory_()

        # Shared input tokens tensor that will be populated with data
        self.input_tokens = torch.zeros(
            (self.cfg.buffer_size_samples, self.cfg.target_ctx_len_toks),
            dtype=torch.int32,
            device=CPU,
        ).share_memory_()

    def start_worker_processes(self):
        """Start worker processes, one per GPU"""
        for proc_idx in range(self.num_processes):
            device = self.devices[proc_idx]
            p = mp.Process(
                target=worker_process,
                args=(
                    proc_idx,
                    device,
                    self.task_queue,
                    self.result_queue,
                    self.cfg,
                    self.input_tokens,
                    self.target_acts,
                    self.target_generated_tokens,
                    self.target_logits,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            logging.info(f"Started worker process {proc_idx} on device {device}")

    def prefetch_next_tokens(self):
        """Start a new thread to fetch the next batch of tokens"""
        # Skip prefetching if using fake tokens
        if self.cfg.use_fake_tokens:
            return
        
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
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def collect_data(self):
        """Coordinate the data collection across multiple GPUs"""
        # Generate random tokens if using fake tokens, otherwise get prefetched tokens
        if self.cfg.use_fake_tokens:
            # Generate random tokens between 0 and vocab_size-1
            toks_init = torch.randint(
                0, 
                self.cfg.vocab_size - 1, 
                (self.cfg.buffer_size_samples, self.cfg.target_ctx_len_toks),
                dtype=torch.int32,
                device=CPU
            )
        else:
            # Get the prefetched tokens
            toks_init = self.token_queue.get()
            # Start prefetching the next batch
            self.prefetch_next_tokens()
            assert toks_init is not None, "Ran out of token data"

        assert toks_init.shape == (
            self.cfg.buffer_size_samples,
            self.cfg.target_ctx_len_toks,
        )

        # Copy tokens to shared memory
        self.input_tokens.copy_(toks_init)

        # Divide data equally across devices
        samples_per_device = self.cfg.buffer_size_samples // self.num_processes

        print("Starting to distribute work to processes")
        print_gpu_memory_usage()
        
        # Distribute work to processes
        submitted_tasks = 0
        for i in range(self.num_processes):
            chunk_start = i * samples_per_device
            # Make sure the last chunk gets any remaining samples
            chunk_end = chunk_start + samples_per_device
            if i == self.num_processes - 1:
                chunk_end = self.cfg.buffer_size_samples
            
            self.task_queue.put((chunk_start, chunk_end))
            submitted_tasks += 1
            logging.info(f"Assigned chunk {chunk_start}:{chunk_end} to worker {i}")

        # Wait for all tasks to complete and check for errors
        completed_tasks = 0
        while completed_tasks < submitted_tasks:
            # Get result (may be success or error)
            result = self.result_queue.get()
            
            # Check if it's an error or success
            if result[0] == "error":
                # It's an error
                error_str = result[1]
                raise RuntimeError(f"Error in worker process: {error_str}")
            
            # It's a success
            _, chunk_start, chunk_end = result
            logging.info(f"Completed chunk {chunk_start}:{chunk_end}")
            completed_tasks += 1
        
        print("All chunks processed successfully")
        print_gpu_memory_usage()

        logging.info(f"All {completed_tasks} chunks processed successfully")

    @property
    def data(self):
        """Return the collected data"""
        return {
            "target_generated_tokens": self.target_generated_tokens,
            "target_logits": self.target_logits,
            "target_acts": self.target_acts,
        }
    
    def terminate_worker_processes(self):
        """Terminate worker processes"""
        for _ in range(len(self.processes)):
            self.task_queue.put(None)
        
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def finish(self):
        """Clean up resources and terminate worker processes"""
        self.terminate_worker_processes()

        # Wait for prefetch thread if it exists
        if not self.cfg.use_fake_tokens and self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=5)

        logging.info("All resources cleaned up")