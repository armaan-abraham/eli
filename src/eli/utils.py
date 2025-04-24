import functools
import logging
import time
from typing import Any, Callable, TypeVar

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def log_gpu_memory_usage(func: F) -> F:
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

        start_time = time.time()
        result = func(*args, **kwargs)

        logger.info(f"Function: {func.__name__}")

        # Log memory usage for each GPU
        for gpu_id in range(num_gpus):
            # Get peak memory usage for this GPU
            peak_memory = torch.cuda.max_memory_allocated(gpu_id)
            used_memory = peak_memory - initial_memories[gpu_id]

            # Convert to more readable format (GB)
            peak_memory_gb = peak_memory / (1024**3)
            used_memory_gb = used_memory / (1024**3)

            logger.info(
                f"GPU {gpu_id} - Peak memory: {peak_memory_gb:.2f} GB, Used memory: {used_memory_gb:.2f} GB"
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

