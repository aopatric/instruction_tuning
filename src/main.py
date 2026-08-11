"""
*TEMP*: Header for Main file.
"""

#
# Imports
#

from model import setup_model
from data import get_loaders
from utils.parser import parse_args
from utils.logging import setup_logger
from utils.misc import seed_everything
from train import eval_model

def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    logger.debug(f"Logging started with level {logger.level}.")


    logger.info(f"Args: {args}")

    seed_everything(args.seed)
    logger.debug(f"Seeded random number generators with value: {args.seed}")

    # load model and tokenizer
    model, tokenizer = setup_model(args.model_name, args.revision)
    logger.info(f"Loaded {args.model_name} at revision {args.revision} with {model.num_parameters()} parameters.")

    # get dataloaders
    train_loader, test_loader = get_loaders(
        tokenizer,
        max_length=args.max_length,
        test_size=args.test_size,
        seed=args.seed,
        batch_size=args.batch_size
    )

    train_eval_loss = eval_model(model, train_loader)

    

if __name__ == "__main__":
    main()