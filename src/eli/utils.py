import functools
import logging
import time
from typing import Any, Callable, TypeVar

import torch
from jaxtyping import Float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def print_gpu_memory_usage_fn(func: F) -> F:
    """
    Decorator that logs the maximum GPU memory usage of a function across all GPUs.
    Only works if CUDA is available.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            logger.info(
                f"CUDA not available, skipping memory logging for {func.__name__}"
            )
            return func(*args, **kwargs)

        # Get number of GPUs
        num_gpus = torch.cuda.device_count()

        # Log initial memory usage for all GPUs
        torch.cuda.reset_peak_memory_stats()
        initial_memories = [torch.cuda.memory_allocated(i) for i in range(num_gpus)]

        result = func(*args, **kwargs)

        logger.info(f"Function: {func.__name__}")

        # Log memory usage for each GPU
        for gpu_id in range(num_gpus):
            # Get peak memory usage for this GPU
            peak_memory = torch.cuda.max_memory_allocated(gpu_id)
            peak_memory_reserved = torch.cuda.max_memory_reserved(gpu_id)

            used_memory = peak_memory - initial_memories[gpu_id]

            # Convert to more readable format (GB)
            peak_memory_gb = peak_memory / (1024**3)
            used_memory_gb = used_memory / (1024**3)
            peak_memory_reserved_gb = peak_memory_reserved / (1024**3)

            logger.info(
                f"GPU {gpu_id} - Peak memory allocated: {peak_memory_gb:.2f} GB, Used memory allocated: {used_memory_gb:.2f} GB, Peak memory reserved: {peak_memory_reserved_gb:.2f} GB"
            )

        return result

    return wrapper


def print_gpu_memory_usage():
    """
    Prints the current GPU memory usage for all available GPUs.
    Only works if CUDA is available.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, cannot print GPU memory usage")
        return

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()

    logger.info("Current GPU Memory Usage:")

    # Log memory usage for each GPU
    for gpu_id in range(num_gpus):
        # Get current memory usage for this GPU
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        reserved_memory = torch.cuda.memory_reserved(gpu_id)

        # Convert to more readable format (GB)
        allocated_memory_gb = allocated_memory / (1024**3)
        reserved_memory_gb = reserved_memory / (1024**3)

        logger.info(
            f"GPU {gpu_id} - Allocated: {allocated_memory_gb:.2f} GB, Reserved: {reserved_memory_gb:.2f} GB"
        )


def calculate_gini(logits: Float[torch.Tensor, "batch tok vocab"]) -> float:
    """Calculate the average Gini coefficient of probabilities over the vocab dimension.

    Args:
        logits: Logits tensor with shape [batch, token, vocab]

    Returns:
        Average Gini coefficient as a float
    """
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Sort probabilities in ascending order
    sorted_probs, _ = torch.sort(probs, dim=-1)

    # Get dimensions
    batch_size, seq_len, vocab_size = probs.shape

    # Create indices for the formula (1 to vocab_size)
    indices = torch.arange(1, vocab_size + 1, device=logits.device, dtype=torch.float32)
    indices = indices.view(1, 1, -1).expand(batch_size, seq_len, -1)

    # Calculate Gini using the formula: G = (2 * sum(i*x_i) / sum(x_i) - (n+1)) / n
    # Where x_i are sorted probabilities and i is the rank from 1 to n
    numerator = 2 * torch.sum(indices * sorted_probs, dim=-1)
    denominator = torch.sum(sorted_probs, dim=-1) * vocab_size
    gini = (numerator / denominator) - (vocab_size + 1) / vocab_size

    # Average over all tokens in the batch
    return gini.mean().item()


def log_decoded_tokens(tokenizer, tokens, file_path, source_name):
    """Decode and log tokens to a file for debugging purposes.

    Args:
        tokenizer: The tokenizer to use for decoding
        tokens: The tokens to decode (first batch entry will be used)
        file_path: Path to the output file
        source_name: Name of the source function/method for context
    """
    try:
        # Get the first batch entry
        token_ids = tokens[0].tolist()

        # Decode the whole sequence first (original functionality)
        decoded_tokens = tokenizer.decode(tokens[0])

        with open(file_path, "w") as f:
            f.write(f"=== Decoded Tokens from {source_name} ===\n")
            f.write(decoded_tokens)
            f.write("\n\n")

            # Add detailed token-by-token analysis
            f.write("=== Token-by-Token Analysis ===\n")
            f.write("Index | Token ID | Is Special | Token Text | Description\n")
            f.write("-" * 80 + "\n")

            # Create a mapping of special token IDs to their names
            special_token_map = {}
            if hasattr(tokenizer, "special_tokens_map"):
                for name, token in tokenizer.special_tokens_map.items():
                    if isinstance(token, str):
                        # Convert token string to ID
                        try:
                            token_id = tokenizer.convert_tokens_to_ids(token)
                            if (
                                token_id != tokenizer.unk_token_id
                            ):  # Make sure it's not the unknown token
                                special_token_map[token_id] = name
                        except Exception:
                            pass

            # Get individual token strings and analyze them
            for i, token_id in enumerate(token_ids):
                # Check if it's a special token
                is_special = token_id in tokenizer.all_special_ids

                # Get the text representation of the token
                token_text = tokenizer.decode([token_id])

                # Try to get the token name or description
                description = ""
                if is_special:
                    if token_id in special_token_map:
                        description = f"Special token: {special_token_map[token_id]}"
                    else:
                        description = "Unknown special token"

                # Write token information
                f.write(
                    f"{i:5d} | {token_id:8d} | {str(is_special):10s} | {repr(token_text):20s} | {description}\n"
                )
    except Exception as e:
        print(f"Error in log_decoded_tokens: {e}")
