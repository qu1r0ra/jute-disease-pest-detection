import random

import numpy as np
import torch
import torch.backends.cudnn


def seed_everything(seed: int):
    """
    Sets the seed for generating random numbers in PyTorch, numpy, and Python's random.
    Also configures CuDNN to be deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")
