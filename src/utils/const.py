"""
Constants for the rest of the codebase.
"""

#
# Imports
#

import logging

#
# Model Support
# 

SUPPORTED_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b"
]

# note: Pythia pre-training checkpoints come in these log-spaced steps
SUPPORTED_REVISIONS = [
    "step0",
    "step1",
    "step2",
    "step4",
    "step8",
    "step16",
    "step32",
    "step64",
    "step128",
    "step256",
    "step512",
    "main"
]

#
# Dataset Support
#

SUPPORTED_DATASETS = [
    "tatsu-lab/alpaca"
]

#
# Logging
#

LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}