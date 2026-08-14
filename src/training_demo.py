"""
*TEMP*: Header for Main file.
"""

#
# Imports
#

import torch

from model import setup_model
from data import get_loaders
from utils.parser import parse_args
from utils.logging import setup_logger
from utils.misc import seed_everything
from train import train

def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    logger.debug(f"Logging started with level {logger.level}.")
    logger.debug(f"Args: {args}")

    seed_everything(args.seed)
    logger.debug(f"Seeded random number generators with value: {args.seed}")

    # check for GPU availability and if it should be used
    device = "cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    logger.info(f"Using device: {device}")

    # load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model(args.model_name, args.revision, args.dtype)
    logger.info(f"Loaded {args.model_name} at revision {args.revision} with {model.num_parameters()} parameters and dtype {args.dtype}.")

    # get dataloaders
    logger.info("Building dataloaders...")
    train_loader, test_loader = get_loaders(
        tokenizer,
        max_length=args.max_length,
        test_size=args.test_size,
        seed=args.seed,
        batch_size=args.batch_size
    )
    logger.info(f"Built train dataloader with {len(train_loader)} batches and test dataloader with {len(test_loader)} batches.")

    logger.info("Performing instruction tuning...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay = 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_loader))
    results = train(
        model = model,
        train_loader = train_loader,
        eval_loader = test_loader,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device,
        num_epochs = args.num_epochs,
        eval_mode = args.eval_mode,
        eval_steps = args.eval_steps,
        eval_epochs = args.eval_epochs,
        max_grad_norm = args.max_grad_norm,
    )
    print(results)

if __name__ == "__main__":
    main()