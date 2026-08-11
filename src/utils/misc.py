"""
Miscellaneous helper functions.
"""

#
# Imports
#

import torch
import numpy as np
import random

#
# Helper functions
#

def seed_everything(seed: int = 42):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed: The seed to use for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)