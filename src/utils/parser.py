"""
Parser for running main.py using command line arguments to simplify rapid experimentation.
"""

#
# Imports
#

import argparse

from utils.const import SUPPORTED_MODELS, SUPPORTED_REVISIONS, LEVEL_MAP, SUPPORTED_DATASETS

def build_parser() -> argparse.ArgumentParser:
    """
    Builds an argument parser for main.py.
    """
    parser = argparse.ArgumentParser(description="Instruction Tuning")

    # misc config
    parser.add_argument_group("Miscellaneous")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators. (Used for reproducibility)")
    parser.add_argument("--log_level", type=str, default="info", choices=LEVEL_MAP.keys(), help="Log level to use.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"], help="Data type to use for the model.")
    parser.add_argument("--no_gpu", action="store_true", default=False, help="Disable GPU usage.")

    # model config
    parser.add_argument_group("Model")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m", choices=SUPPORTED_MODELS, help="Model name to use.")
    parser.add_argument("--revision", type=str, default="step0", choices=SUPPORTED_REVISIONS, help="Revision of the model to use.")

    # dataset config
    parser.add_argument_group("Dataset")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of sequences.")
    parser.add_argument("--test_size", type=float, default=0.8, help="Proportion of the dataset (or number of samples) to include in the test split.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataloaders.")

    # training config
    parser.add_argument_group("Training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--eval_mode", type=str, default="epoch", choices=["epoch", "step", "all"], help="Mode to use for evaluation.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations.")
    parser.add_argument("--eval_epochs", type=int, default=1, help="Number of epochs between evaluations.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping.")

    return parser

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parses the arguments for main.py.
    """
    parser = build_parser()
    return parser.parse_args(args)