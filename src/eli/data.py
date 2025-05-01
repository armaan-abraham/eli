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
from jaxtyping import Int
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.config import CPU, Config, cfg
from eli.utils import print_gpu_memory_usage_fn

# Setup directories
this_dir = Path(__file__).parent
cache_dir = this_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)


def load_tokenizer():
    # Handles custom schlepping and idiosyncracies for loading the currently specified tok
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name)
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
        assert isinstance(
            text[0][0], str
        ), f"Expected list of lists of strings, got {type(text[0][0])}"

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

        assert (
            num_tokens > seq_len
        ), f"Num tokens: {num_tokens} is less than seq_len: {seq_len}"
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


def stream_training_chunks(cfg: Config = cfg):
    """
    Stream tokenized chunks from a dataset for training.

    Args:
        cfg: Configuration object

    Yields:
        Batches of tokenized text as tensors
    """
    CLIENT_TIMEOUT_SECONDS = 60 * 60 * 2
    storage_options = {
        "client_kwargs": {
            "timeout": aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_SECONDS),
        }
    }

    # Load and prepare dataset
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

    # Tokenize the dataset
    tokenizer = load_tokenizer()
    dataset_iter = tokenize_and_concatenate(
        dataset_iter,
        tokenizer,
        streaming=True,
        max_length=cfg.target_ctx_len_toks,
        add_bos_token=True,
        column_name=cfg.dataset_column_name,
    )

    dataset_iter = dataset_iter.batch(cfg.buffer_size_samples)

    for batch in dataset_iter:
        yield batch["tokens"].to(dtype=torch.int32, device="cpu")


def _process_batch(
    batch_toks: Int[torch.Tensor, "batch tok"],
    batch_start: int,
    batch_end: int,
    target_model: AutoModelForCausalLM,
    target_model_act_collection: transformer_lens.HookedTransformer,
    tokenizer: AutoTokenizer,
    target_acts: torch.Tensor,
    target_generated_tokens: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> None:
    """
    Process a single batch of tokens through the models.

    Args:
        batch_toks: Input tokens tensor
        batch_start, batch_end: Start and end indices of this batch
        target_model: Model for token generation
        target_model_act_collection: Model for activation collection
        tokenizer: Tokenizer for the models
        target_acts, target_generated_tokens: Output tensors
        cfg: Configuration object
        device: Device to run computation on
    """
    # Use autocast for model operations
    with torch.autocast(device_type=device.type, dtype=cfg.dtype):
        # Collect activations
        _, cache = target_model_act_collection.run_with_cache(
            batch_toks,
            # stop_at_layer=cfg.layer + 1,
            # names_filter=cfg.act_name,
            return_cache_object=True,
        )

        # Get activations and move to shared memory
        acts = cache["normalized"][:, -cfg.target_acts_collect_len_toks:, :] # [batch tok d_model]
        acts_cat = einops.rearrange(acts, "batch tok d_model -> batch (tok d_model)")
        target_acts[batch_start:batch_end] = acts_cat.cpu()

        # Generate tokens
        length_toks = cfg.target_ctx_len_toks + cfg.target_generation_len_toks
        attention_mask = torch.ones_like(batch_toks, dtype=torch.int32)

        # Generate tokens
        batch_toks_with_gen = target_model.generate(
            batch_toks,
            attention_mask=attention_mask,
            max_length=length_toks,
            min_length=length_toks,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Store generated tokens in shared memory
        generated_tokens = batch_toks_with_gen[:, -cfg.target_generation_len_toks :]
        target_generated_tokens[batch_start:batch_end] = generated_tokens.cpu()


def process_data_chunk(
    chunk_start: int,
    chunk_end: int,
    input_tokens: torch.Tensor,
    target_acts: torch.Tensor,
    target_generated_tokens: torch.Tensor,
    cfg: Config,
    device: torch.device,
    target_model: Optional[AutoModelForCausalLM] = None,
    target_model_act_collection: Optional[transformer_lens.HookedTransformer] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> None:
    """
    Process a chunk of data through the models.

    Args:
        chunk_start, chunk_end: Start and end indices of the chunk to process
        input_tokens, target_acts, target_generated_tokens: Data tensors
        cfg: Configuration object
        device: Device to run computation on
        target_model: Model for token generation (loaded if None)
        target_model_act_collection: Model for activation collection (loaded if None)
        tokenizer: Tokenizer for the models (loaded if None)
    """
    # Load models if not provided
    models_loaded_here = False
    if target_model is None or target_model_act_collection is None or tokenizer is None:
        models_loaded_here = True
        tokenizer = load_tokenizer()
        target_model = AutoModelForCausalLM.from_pretrained(cfg.target_model_name).to(
            device
        )
        target_model_act_collection = (
            transformer_lens.HookedTransformer.from_pretrained(
                cfg.target_model_name
            ).to(device)
        )

    logging.info(f"Processing chunk {chunk_start}:{chunk_end} on {device}")

    # Process the chunk in batches
    batch_size = cfg.target_model_batch_size_samples
    for batch_start in range(chunk_start, chunk_end, batch_size):
        batch_end = min(batch_start + batch_size, chunk_end)
        logging.info(f"Processing batch {batch_start}:{batch_end}")

        # Get batch tokens and process
        batch_toks = input_tokens[batch_start:batch_end].to(device)
        _process_batch(
            batch_toks,
            batch_start,
            batch_end,
            target_model,
            target_model_act_collection,
            tokenizer,
            target_acts,
            target_generated_tokens,
            cfg,
            device,
        )

    # Clean up models if we loaded them
    if models_loaded_here:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def worker_process(
    proc_idx: int,
    device: torch.device,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    cfg: Config,
    input_tokens: torch.Tensor,
    target_acts: torch.Tensor,
    target_generated_tokens: torch.Tensor,
) -> None:
    """
    Worker process that loads models and processes batches.

    Args:
        proc_idx: Process index
        device: Device to run computation on
        task_queue: Queue to get tasks from
        result_queue: Queue to report results
        cfg: Configuration object
        input_tokens, target_acts, target_generated_tokens: Shared tensors
    """
    try:
        print("proc_idx", proc_idx, "setting CUDA_VISIBLE_DEVICES", device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(proc_idx)
        # Now that we set the visible device, we no longer reference it with an
        # ordinal
        device = torch.device("cuda")

        # Load models in the worker process
        tokenizer = load_tokenizer()
        target_model = AutoModelForCausalLM.from_pretrained(cfg.target_model_name).to(
            CPU
        )
        target_model_act_collection = (
            transformer_lens.HookedTransformer.from_pretrained(
                cfg.target_model_name,
                device=CPU,
                n_devices=1,
            ).to(CPU)
        )

        logging.info(f"Worker {proc_idx} loaded models on {device}")

        while True:
            # Get chunk range from the queue
            chunk_range = task_queue.get()
            if chunk_range is None:  # Sentinel value to stop
                break

            torch.cuda.reset_peak_memory_stats()

            # Move models to device for processing
            target_model.to(device)
            target_model_act_collection.to(device)

            chunk_start, chunk_end = chunk_range

            # Process the chunk using the helper function
            process_data_chunk(
                chunk_start,
                chunk_end,
                input_tokens,
                target_acts,
                target_generated_tokens,
                cfg,
                device,
                target_model,
                target_model_act_collection,
                tokenizer,
            )

            # Move models back to CPU and clear GPU memory
            target_model.to(CPU)
            target_model_act_collection.to(CPU)
            if device.type == "cuda":
                torch.cuda.empty_cache()

            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            max_reserved = torch.cuda.max_memory_reserved() / (1024**3)  # GB
            logging.info(
                f"Worker {proc_idx} processed chunk {chunk_start}:{chunk_end}, "
                f"Max GPU memory allocated: {max_allocated:.2f} GB, "
                f"Max GPU memory reserved: {max_reserved:.2f} GB"
            )

            # Notify completion of the chunk
            result_queue.put(("success", chunk_start, chunk_end))

        logging.info(f"Worker {proc_idx} finished")

        # Final cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        # Send error back to main process
        error_str = f"Error in worker {proc_idx}: {str(e)}\n{traceback.format_exc()}"
        result_queue.put(("error", error_str))
        logging.error(error_str)

        # Clear cache on error
        if device.type == "cuda":
            torch.cuda.empty_cache()


class DataCollector:
    """
    Manages the process of collecting data from models across multiple devices.

    Handles token streaming, worker processes, and data aggregation.
    """

    def __init__(self, cfg: Config = cfg):
        """
        Initialize the data collector.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.tokenizer = load_tokenizer()
        self.use_workers = cfg.use_data_collector_workers

        # Setup token streaming
        self._setup_token_streaming()

        # Setup multiprocessing
        if self.use_workers:
            self._setup_multiprocessing()

        # Create shared memory tensors
        self.setup_shared_tensors()

        # Start prefetching tokens
        self.prefetch_next_tokens()

        # Start worker processes
        if self.use_workers:
            self.start_worker_processes()

    def _setup_token_streaming(self):
        """Configure token streaming based on configuration"""
        if not self.cfg.use_fake_tokens:
            self.token_stream = stream_training_chunks(self.cfg)
            self.token_queue = Queue(maxsize=1)
            self.prefetch_thread = None
        else:
            logging.info("Using fake tokens for testing")
            self.token_stream = None
            self.token_queue = None
            self.prefetch_thread = None

    def _setup_multiprocessing(self):
        """Configure multiprocessing resources"""
        mp.set_start_method("spawn", force=True)

        # Detect available GPUs
        self.num_gpus = torch.cuda.device_count()
        self.num_processes = self.num_gpus if self.num_gpus > 0 else 1
        self.devices = (
            [torch.device(f"cuda:{i}") for i in range(self.num_gpus)]
            if self.num_gpus > 0
            else [torch.device("cpu")]
        )

        logging.info(f"Using {self.num_processes} processes on devices: {self.devices}")

        # Initialize multiprocessing components
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = []

    def setup_shared_tensors(self):
        """Create shared memory tensors for input and output data"""
        # Calculate and log memory sizes before initialization
        target_generated_tokens_size = (
            self.cfg.buffer_size_samples
            * self.cfg.target_generation_len_toks
            * 4  # int32 = 4 bytes
        )
        target_acts_size = (
            self.cfg.buffer_size_samples
            * self.cfg.target_model_agg_acts_dim
            * 4  # float32 = 4 bytes
        )
        input_tokens_size = (
            self.cfg.buffer_size_samples
            * self.cfg.target_ctx_len_toks
            * 4  # int32 = 4 bytes
        )

        # Log sizes in more readable format (bytes, KB, MB, GB)
        logging.info(
            f"target_generated_tokens size: {target_generated_tokens_size} bytes "
            f"({target_generated_tokens_size/1024/1024:.2f} MB)"
        )
        logging.info(
            f"target_acts size: {target_acts_size} bytes "
            f"({target_acts_size/1024/1024:.2f} MB)"
        )
        logging.info(
            f"input_tokens size: {input_tokens_size} bytes "
            f"({input_tokens_size/1024/1024:.2f} MB)"
        )
        logging.info(
            f"Total shared memory size: "
            f"{(target_generated_tokens_size + target_acts_size + input_tokens_size)/1024**3:.2f} GB"
        )

        # Define shared memory tensors with proper dtype for storage
        self.target_generated_tokens = torch.zeros(
            (self.cfg.buffer_size_samples, self.cfg.target_generation_len_toks),
            dtype=torch.int32,
            device=CPU,
        ).share_memory_()

        self.target_acts = torch.zeros(
            (self.cfg.buffer_size_samples, self.cfg.target_model_agg_acts_dim),
            dtype=torch.float32,
            device=CPU,
        ).share_memory_()

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
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            logging.info(f"Started worker process {proc_idx} on device {device}")

    def prefetch_next_tokens(self):
        """Start a new thread to fetch the next batch of tokens"""
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
                # Handle end of dataset
                self.token_queue.put(None)

        self.prefetch_thread = threading.Thread(target=fetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def _load_tokens(self) -> torch.Tensor:
        """
        Load tokens into shared memory, either generating fake tokens
        or fetching real ones from the token stream.

        Returns:
            The loaded tokens (also stored in self.input_tokens)
        """
        # Get or generate tokens
        if self.cfg.use_fake_tokens:
            # Generate random tokens for testing
            toks_init = torch.randint(
                0,
                self.cfg.vocab_size_target - 1,
                (self.cfg.buffer_size_samples, self.cfg.target_ctx_len_toks),
                dtype=torch.int32,
                device=CPU,
            )
        else:
            # Get prefetched tokens
            toks_init = self.token_queue.get()
            self.prefetch_next_tokens()
            assert toks_init is not None, "Ran out of token data"

        # Verify token shape
        assert toks_init.shape == (
            self.cfg.buffer_size_samples,
            self.cfg.target_ctx_len_toks,
        )

        # Copy tokens to shared memory
        self.input_tokens.copy_(toks_init)

        return toks_init

    def collect_data(self):
        """Coordinate data collection across multiple GPUs"""
        # Load tokens into shared memory
        self._load_tokens()

        # Distribute work to processes
        if self.use_workers:
            self._distribute_work()
        else:
            # Process data directly in the main process
            device = self.cfg.device
            logging.info(f"Processing data directly on {device} without workers")

            # Process the entire buffer as a single chunk
            process_data_chunk(
                0,
                self.cfg.buffer_size_samples,
                self.input_tokens,
                self.target_acts,
                self.target_generated_tokens,
                self.cfg,
                device,
            )

            logging.info("Direct data processing completed")

    def _distribute_work(self):
        """Distribute work among workers and wait for completion"""
        samples_per_device = self.cfg.buffer_size_samples // self.num_processes
        submitted_tasks = 0

        # Assign chunks to workers
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
            result = self.result_queue.get()

            # Check if it's an error or success
            if result[0] == "error":
                # Handle error
                error_str = result[1]
                raise RuntimeError(f"Error in worker process: {error_str}")

            # Handle successful completion
            _, chunk_start, chunk_end = result
            logging.info(f"Completed chunk {chunk_start}:{chunk_end}")
            completed_tasks += 1

        logging.info(f"All {completed_tasks} chunks processed successfully")

    @property
    def data(self):
        """Return the collected data"""
        return {
            "target_generated_tokens": self.target_generated_tokens,
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
        if self.use_workers:
            self.terminate_worker_processes()

        # Wait for prefetch thread if it exists
        if (
            not self.cfg.use_fake_tokens
            and self.prefetch_thread
            and self.prefetch_thread.is_alive()
        ):
            self.prefetch_thread.join(timeout=5)

        logging.info("All resources cleaned up")
