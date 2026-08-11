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
    parser.add_argument("--log_level", type=str, default="debug", choices=LEVEL_MAP.keys(), help="Log level to use.")

    # model config
    parser.add_argument_group("Model")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m", choices=SUPPORTED_MODELS, help="Model name to use.")
    parser.add_argument("--revision", type=str, default="main", choices=SUPPORTED_REVISIONS, help="Revision of the model to use.")

    # dataset config
    parser.add_argument_group("Dataset")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of sequences.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataloaders.")

    return parser

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parses the arguments for main.py.
    """
    parser = build_parser()
    return parser.parse_args(args)