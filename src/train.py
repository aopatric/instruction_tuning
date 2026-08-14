"""
*TEMP*: Header for Training file.
"""

#
# Imports
#

import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Literal

#
# Core Training Functions
#

@torch.no_grad
def do_eval_epoch(model: torch.nn.Module, eval_loader: DataLoader, device: torch.device):
    """
    *TEMP* docstring, returns mean CE loss per supervised token across the eval
    set, along with top-1 next-token accuracy over those same tokens.
    """

    was_training = model.training

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct = 0
    
    for batch in tqdm(eval_loader, desc="Evaluating model..."):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # the model shifts internally, so scoring position i against label i+1
        shift_labels = labels[:, 1:]
        supervised = shift_labels != -100
        num_tokens = supervised.sum().item()

        predictions = outputs.logits[:, :-1].argmax(dim=-1)
        correct += ((predictions == shift_labels) & supervised).sum().item()

        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    model.train(was_training)
    return total_loss / total_tokens, correct / total_tokens * 100.0

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    num_epochs: int = 5,
    eval_mode: str  = "all",
    eval_steps: int = 100,
    eval_epochs: int = 1,
    max_grad_norm: float = 1.0,
):
    """
    *TEMP* docstring
    """

    model.to(device)
    model.train()

    results = {
        "eval_results": dict(),
        "train_results": dict(),
    }

    total_steps = 0

    for epoch in tqdm(range(num_epochs), desc="Training model..."):
        epoch_loss = 0.0
        correct = 0
        epoch_tokens = 0

        for batch in (pbar:= tqdm(train_loader)):
            total_steps += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

            shift_labels = labels[:, 1:]
            supervised = shift_labels != -100
            num_tokens = supervised.sum().item()

            predictions = outputs.logits[:, :-1].argmax(dim=-1)
            correct += ((predictions == shift_labels) & supervised).sum().item()

            epoch_loss += outputs.loss.item() * num_tokens
            epoch_tokens += num_tokens

            pbar.set_postfix({"loss": epoch_loss / epoch_tokens, "accuracy": correct / epoch_tokens * 100.0})

            if eval_mode in ["step", "all"] and total_steps % eval_steps == 0:
                loss, accuracy = do_eval_epoch(model, eval_loader, device)
                results["eval_results"][f"step{total_steps}"] = {"loss": loss, "accuracy": accuracy}
                print(f"Eval results at step {total_steps}: loss {loss:.4f}, accuracy {accuracy:.2f}%")
        
        results["train_results"][f"epoch{epoch+1}"] = {"loss": epoch_loss / epoch_tokens, "accuracy": correct / epoch_tokens * 100.0}

        if eval_mode in ["epoch", "all"] and (epoch + 1) % eval_epochs == 0:
            loss, accuracy = do_eval_epoch(model, eval_loader, device)
            results["eval_results"][f"epoch{epoch+1}"] = {"loss": loss, "accuracy": accuracy}

    return results