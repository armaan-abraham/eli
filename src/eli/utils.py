import functools
import time
from typing import Callable, TypeVar, Any

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

def log_gpu_memory_usage(func: F) -> F:
    """
    Decorator that logs the maximum GPU memory usage of a function.
    Only works if CUDA is available.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            logger.info(f"CUDA not available, skipping memory logging for {func.__name__}")
            return func(*args, **kwargs)
        
        # Log initial memory usage
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        
        # Get peak memory usage
        peak_memory = torch.cuda.max_memory_allocated()
        used_memory = peak_memory - initial_memory
        
        # Convert to more readable format (MB)
        peak_memory_gb = peak_memory / (1024 ** 3)
        used_memory_gb = used_memory / (1024 ** 3)
        
        logger.info(f"Function: {func.__name__}")
        logger.info(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
        logger.info(f"Used GPU memory: {used_memory_gb:.2f} GB")
        
        return result
    
    return wrapper 