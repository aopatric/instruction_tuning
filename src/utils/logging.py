"""
Scaffolding for logging including command line debug and tensorboard.
"""

#
# Imports
#

import logging

from utils.const import LEVEL_MAP

#
# Core Command Line Logging
#

def setup_logger(level: str = "info") -> logging.Logger:
    """
    Sets up the logger for the application.

    Args:
        level: The level of logging to use.
    """
    logging.basicConfig(level=LEVEL_MAP.get(level, logging.INFO))
    return logging.getLogger(__name__)