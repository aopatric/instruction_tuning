"""
*TEMP*: Header for Data file.
"""

#
# Imports
#

import torch

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPTNeoXTokenizer
from typing import Any, Dict, Optional

#
# Core Dataloading Functions
#

def get_loaders(
    tokenizer: GPTNeoXTokenizer,
    max_length: int = 1024,
    test_size: float = 0.2,
    seed: int = 42,
    batch_size: int = 8,
):
    """
    *TEMP* docstring
    """

    # load from huggingface
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    tokenized_data = dataset.map(
        lambda example: tokenize_example(example, tokenizer, max_length),
        remove_columns=dataset.column_names
    )

    splits = tokenized_data.train_test_split(test_size=test_size, seed=seed)
    train, test = splits["train"], splits["test"]
    collator = InstructionCollator(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    return train_loader, test_loader
    

def tokenize_example(example: Dict[str, Any], tokenizer: GPTNeoXTokenizer, max_length: int):
    full_example = example["text"] + tokenizer.eos_token

    response_idx = full_example.find("### Response") + len("### Response")
    prompt_only = full_example[:response_idx]

    all_token_ids = tokenizer(
        full_example,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_only,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    labels = all_token_ids.copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    return {
        "input_ids": all_token_ids,
        "labels": labels
    }

class InstructionCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        max_len = max(len(example["input_ids"]) for example in batch)

        if self.pad_to_multiple_of is not None:
            assert self.pad_to_multiple_of >= 1, "pad_to_multiple_of must be >= 1"
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        input_ids, labels, attention_mask = [], [], []

        for example in batch:
            padding = max_len - len(example['input_ids'])
            input_ids.append(example['input_ids'] + [self.pad_token_id] * padding)
            labels.append(example['labels'] + [-100] * padding)
            attention_mask.append([1] * len(example['input_ids']) + [0] * padding)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(attention_mask)
        }
