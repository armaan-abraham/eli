import logging
import os
import time
import traceback

import torch
import torch.multiprocessing as mp
import transformer_lens
from transformers import AutoModelForCausalLM, AutoTokenizer

from eli.datasets.config import DatasetConfig, ds_cfg

CPU = torch.device("cpu")

"""Configure logging to output to stdout with appropriate formatting."""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to stdout
    ],
)


def load_tokenizer():
    """Load the tokenizer for the target model"""
    tokenizer = AutoTokenizer.from_pretrained(ds_cfg.target_model_name)
    return tokenizer


def filter_empty_generations(
    tokenizer: AutoTokenizer,
    target_acts: torch.Tensor,
    target_generated_tokens: torch.Tensor,
):
    eos_token = tokenizer.eos_token_id

    # Filter out rows with only EOS tokens
    eos_mask = torch.any(target_generated_tokens != eos_token, dim=1)
    filtered_tokens = target_generated_tokens[eos_mask]
    filtered_acts = target_acts[eos_mask]

    # Filter out rows with only whitespace
    # Use batch_decode for efficiency and check each text for non-whitespace content
    decoded_texts = tokenizer.batch_decode(filtered_tokens)
    content_mask = torch.tensor(
        [len(text.strip()) > 0 for text in decoded_texts], dtype=torch.bool
    )

    target_generated_tokens = filtered_tokens[content_mask]
    target_acts = filtered_acts[content_mask]

    return target_acts, target_generated_tokens


def process_tokens_with_model(
    batch_toks: torch.Tensor,
    batch_start: int,
    batch_end: int,
    target_model: AutoModelForCausalLM,
    target_model_act_collection: transformer_lens.HookedTransformer,
    tokenizer: AutoTokenizer,
    target_acts: torch.Tensor,
    target_generated_tokens: torch.Tensor,
    device: torch.device,
):
    """
    Process a batch of tokens through the target models to collect activations and generations.

    Args:
        batch_toks: Input tokens tensor
        batch_start, batch_end: Start and end indices of this batch
        target_model: Model for token generation
        target_model_act_collection: Model for activation collection
        tokenizer: Tokenizer for the models
        target_acts, target_generated_tokens: Output tensors
        device: Device to run computation on
    """
    # Use autocast for model operations
    with torch.autocast(device_type=device.type, dtype=ds_cfg.dtype):
        # Produce activations
        _, cache = target_model_act_collection.run_with_cache(
            batch_toks,
            stop_at_layer=ds_cfg.target_acts_layer_range[1] + 1,
            return_cache_object=True,
        )

        # Extract activations and move to shared memory
        acts = torch.stack(
            [
                cache.cache_dict[
                    transformer_lens.utils.get_act_name(ds_cfg.site, layer)
                ][:, -ds_cfg.target_acts_collect_len_toks :, :]  # [batch tok d_model]
                for layer in range(
                    ds_cfg.target_acts_layer_range[0],
                    ds_cfg.target_acts_layer_range[1] + 1,
                )
            ],
            dim=2,
        )  # [batch tok layer d_model]

        del cache

        assert acts.shape == (
            ds_cfg.target_model_batch_size_samples,
            ds_cfg.target_acts_collect_len_toks,
            ds_cfg.target_acts_layer_range[1] - ds_cfg.target_acts_layer_range[0] + 1,
            ds_cfg.target_model_act_dim,
        )
        target_acts[batch_start:batch_end] = acts.cpu()

        # Generate tokens
        length_toks = ds_cfg.target_ctx_len_toks + ds_cfg.target_generation_len_toks
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
        generated_tokens = batch_toks_with_gen[:, -ds_cfg.target_generation_len_toks :]
        target_generated_tokens[batch_start:batch_end] = generated_tokens.cpu()


def worker_process(
    proc_idx: int,
    device: torch.device,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    target_acts: torch.Tensor,
    target_generated_tokens: torch.Tensor,
    input_tokens: torch.Tensor,
):
    """
    Worker process that loads models and processes batches.

    Args:
        proc_idx: Process index
        device: Device to run computation on
        task_queue: Queue to get tasks from
        result_queue: Queue to report results
        target_acts, target_generated_tokens: Shared tensors
    """
    try:
        logging.info(f"Worker {proc_idx} starting on device {device}")
        if device.type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(proc_idx)
            # Now we use cuda:0 since we've set the visible device
            device = torch.device("cuda")

        # Load models in the worker process - directly to the device
        tokenizer = load_tokenizer()
        target_model = AutoModelForCausalLM.from_pretrained(
            ds_cfg.target_model_name
        ).to(device)
        target_model_act_collection = (
            transformer_lens.HookedTransformer.from_pretrained(
                ds_cfg.target_model_name, device=device, n_devices=1
            )
        )

        logging.info(f"Worker {proc_idx} loaded models on {device}")

        while True:
            # Get task from queue
            task = task_queue.get()
            if task is None:  # Sentinel value to stop
                break

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            chunk_start, chunk_end = task
            batch_size = ds_cfg.target_model_batch_size_samples

            # Process the chunk in batches
            for batch_start in range(chunk_start, chunk_end, batch_size):
                batch_end = min(batch_start + batch_size, chunk_end)
                logging.info(
                    f"Worker {proc_idx} processing batch {batch_start}:{batch_end}"
                )

                # Get batch tokens and process
                batch_toks = input_tokens[batch_start:batch_end].to(device)
                process_tokens_with_model(
                    batch_toks,
                    batch_start,
                    batch_end,
                    target_model,
                    target_model_act_collection,
                    tokenizer,
                    target_acts,
                    target_generated_tokens,
                    device,
                )

            # Notify completion of the chunk
            result_queue.put(("success", chunk_start, chunk_end))

            if device.type == "cuda":
                logging.info(
                    f"Worker {proc_idx} finished with peak memory {torch.cuda.max_memory_allocated() / 1024**3} GB, peak memory reserved {torch.cuda.max_memory_reserved() / 1024**3} GB"
                )

        logging.info(f"Worker {proc_idx} finished")

    except Exception as e:
        # Send error back to main process
        error_str = f"Error in worker {proc_idx}: {str(e)}\n{traceback.format_exc()}"
        result_queue.put(("error", error_str))
        logging.error(error_str)


class TargetDataStream:
    """
    Processes token data across multiple GPUs to collect activations and generations.
    Returns a stream of batches with the processed data.
    """

    def __init__(self, token_stream, ds_cfg: DatasetConfig = ds_cfg):
        """
        Initialize the target data stream.

        Args:
            token_stream: Iterator yielding batches of token tensors
        """
        self.ds_cfg = ds_cfg

        self.token_stream = token_stream
        self.tokenizer = load_tokenizer()

        # Setup multiprocessing
        mp.set_start_method("spawn", force=True)
        self.num_gpus = torch.cuda.device_count()
        self.num_processes = self.num_gpus if self.num_gpus > 0 else 1
        self.devices = (
            [torch.device(f"cuda:{i}") for i in range(self.num_gpus)]
            if self.num_gpus > 0
            else [torch.device("cpu")]
        )

        logging.info(f"Using {self.num_processes} processes on devices: {self.devices}")

        # Setup multiprocessing components
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = []

        self._setup_shared_tensors()

        # Start worker processes
        self._start_worker_processes()

        # Initialize buffers for each output type
        self.buffer = {"target_acts": [], "target_generated_tokens": []}
        self.buffer_size = 0

    def _start_worker_processes(self):
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
                    self.target_acts,
                    self.target_generated_tokens,
                    self.input_tokens,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            logging.info(f"Started worker process {proc_idx} on device {device}")

    def _setup_shared_tensors(self):
        """Create shared memory tensors for output data with given batch size"""

        # --- Result tensors ---
        self.target_generated_tokens = torch.zeros(
            (
                self.ds_cfg.dataset_entry_size_samples,
                self.ds_cfg.target_generation_len_toks,
            ),
            dtype=torch.int32,
            device=CPU,
        ).share_memory_()

        self.target_acts = torch.zeros(
            (
                self.ds_cfg.dataset_entry_size_samples,
                self.ds_cfg.target_acts_collect_len_toks,
                self.ds_cfg.target_acts_layer_range[1]
                - self.ds_cfg.target_acts_layer_range[0]
                + 1,
                self.ds_cfg.target_model_act_dim,
            ),
            dtype=self.ds_cfg.act_storage_dtype,
            device=CPU,
        ).share_memory_()

        # --- Input tensors ---
        # This is a shared tensor to avoid passing input data via queues, which is slow
        self.input_tokens = torch.zeros(
            (self.ds_cfg.dataset_entry_size_samples, self.ds_cfg.target_ctx_len_toks),
            dtype=torch.int32,
            device=CPU,
        ).share_memory_()

    def collect_model_outputs(self, tokens_batch):
        """
        Process a batch of tokens across multiple GPUs to collect activations and generations.

        Args:
            tokens_batch: Tensor of tokens [batch_size, ctx_len]

        Returns:
            Dictionary with processed tensors
        """
        atom_size = self.ds_cfg.dataset_entry_size_samples
        assert (
            tokens_batch.shape[0] == atom_size
        ), f"Token batch size {tokens_batch.shape[0]} does not match expected size {atom_size}"

        # Copy tokens to shared memory
        self.input_tokens[:] = tokens_batch

        # Distribute work to processes
        samples_per_device = atom_size // self.num_processes
        submitted_tasks = 0

        # Assign chunks to workers
        for i in range(self.num_processes):
            chunk_start = i * samples_per_device
            # Make sure the last chunk gets any remaining samples
            chunk_end = chunk_start + samples_per_device
            if i == self.num_processes - 1:
                chunk_end = atom_size

            if chunk_start < chunk_end:  # Only submit non-empty chunks
                self.task_queue.put((chunk_start, chunk_end))
                submitted_tasks += 1
                logging.info(f"Assigned chunk {chunk_start}:{chunk_end} to worker {i}")

        # Wait for all tasks to complete and check for errors
        self._wait_for_workers(submitted_tasks)

        target_acts, target_generated_tokens = filter_empty_generations(
            self.tokenizer,
            self.target_acts.clone(),
            self.target_generated_tokens.clone(),
        )

        # Return the processed data
        return {
            "target_acts": target_acts,
            "target_generated_tokens": target_generated_tokens,
        }

    def _wait_for_workers(self, expected_tasks=None):
        """
        Wait for workers to complete their tasks and handle any errors.

        Args:
            expected_tasks: Number of expected task completions
        """
        completed_tasks = 0
        while expected_tasks is None or completed_tasks < expected_tasks:
            if expected_tasks is None and self.result_queue.empty():
                # If we're just waiting for in-progress tasks and queue is empty, we're done
                break

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

            if expected_tasks is not None:
                logging.info(f"Completed {completed_tasks}/{expected_tasks} tasks")

        if expected_tasks is not None:
            logging.info(f"All {completed_tasks} chunks processed successfully")

    def __iter__(self):
        """Return iterator for the stream"""
        return self

    def __next__(self):
        """
        Get the next batch of processed data with exactly dataset_entry_size_samples.
        Uses a buffer to accumulate samples until we have enough.
        """
        try:
            # Keep collecting data until we have a full batch
            while self.buffer_size < self.ds_cfg.dataset_entry_size_samples:
                # Get next batch of tokens
                tokens_batch = next(self.token_stream)
                result = self.collect_model_outputs(tokens_batch)

                # Add results to buffer
                self.buffer["target_acts"].append(result["target_acts"])
                self.buffer["target_generated_tokens"].append(
                    result["target_generated_tokens"]
                )
                self.buffer_size += result["target_acts"].shape[0]

            # Concatenate buffer data
            target_acts = torch.cat(self.buffer["target_acts"], dim=0)
            target_generated_tokens = torch.cat(
                self.buffer["target_generated_tokens"], dim=0
            )

            # Extract exactly stream_batch_size_samples
            result = {
                "target_acts": target_acts[: self.ds_cfg.dataset_entry_size_samples],
                "target_generated_tokens": target_generated_tokens[
                    : self.ds_cfg.dataset_entry_size_samples
                ],
            }

            # Keep remaining samples in buffer
            self.buffer["target_acts"] = [
                target_acts[self.ds_cfg.dataset_entry_size_samples :]
            ]
            self.buffer["target_generated_tokens"] = [
                target_generated_tokens[self.ds_cfg.dataset_entry_size_samples :]
            ]
            self.buffer_size = target_acts[
                self.ds_cfg.dataset_entry_size_samples :
            ].shape[0]

            return result

        except StopIteration:
            # If we have partial data in the buffer but not enough for a full batch
            if 0 < self.buffer_size < self.ds_cfg.dataset_entry_size_samples:
                logging.warning(
                    f"End of stream reached with {self.buffer_size} samples in buffer, "
                    f"which is less than the required {self.ds_cfg.dataset_entry_size_samples}"
                )

            # Clean up and re-raise StopIteration
            self.close()
            raise

    def close(self):
        """Clean up resources"""
        # Signal workers to stop
        for _ in range(len(self.processes)):
            self.task_queue.put(None)

        # Wait for workers to finish
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        logging.info("All worker processes terminated")


def stream_target_data(token_stream):
    """
    Stream target model activations and generations from a token stream.

    Args:
        token_stream: Iterator yielding batches of token tensors

    Returns:
        Iterator yielding dictionaries with processed tensors
    """
    return TargetDataStream(token_stream)
