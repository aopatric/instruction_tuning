"""
*TEMP*: Header for Models file.
"""

#
# Imports
#

import torch

from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizer
from typing import Tuple
from utils.const import SUPPORTED_MODELS, SUPPORTED_REVISIONS

#
# Core Model Components
#

def setup_model(model_name: str, revision: str = "main", dtype=torch.float16) -> Tuple[GPTNeoXForCausalLM, GPTNeoXTokenizer]:
    """
    Sets up the given model for use in experimentation. Returns the selected
    model and tokenizer as a tuple (model, tokenizer).

    Args:
        model_name: The name of the model to set up (e.g. "bert").
        revision: The revision of the model to use. (e.g. "step1", "step512", "main")

    Returns:
        A tuple containing the model and tokenizer (model, tokenizer).

    Raises:
        ValueError: If the model_name is not found.
    """

    assert model_name in SUPPORTED_MODELS, f"Model '{model_name}' is not supported."
    assert revision in SUPPORTED_REVISIONS, f"Revision '{revision}' is not supported for model '{model_name}'."

    tokenizer = GPTNeoXTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=revision, dtype=dtype)

    return model, tokenizer