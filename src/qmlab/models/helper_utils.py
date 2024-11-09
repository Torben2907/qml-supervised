"""General helper functions, e.g. setting seed of random number generation, 
or choosing computational device (CUDA, MPS, CPU, ...)
"""

import torch
import logging


def get_device_torch() -> str:
    """This function is gonna check if GPU is available.
    Fall back to CPU usage when no GPU is available.

    Returns:
        str: of device to use for computation with PyTorch.
    """
    if torch.backends.cuda.is_built():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        logging.warning("No GPU found. Falling back to CPU usage.")
        return "cpu"
