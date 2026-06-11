"""
Path adapter for backward compatibility
========================================

This module provides backward-compatible path variables
for scripts that were using hardcoded paths.
"""

from pathlib import Path
import sys

# Add src to path
SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

# Import all path constants
from gaokao_recommender.paths import *

# Backward-compatible aliases
PROJECT_ROOT = PROJECT_ROOT  # Already defined in paths
DATA_DIR = DATA_RAW_DIR  # Old scripts used DATA_DIR for raw data
OUTPUT_DIR = DATA_PROCESSED_DIR  # Old scripts used OUTPUT_DIR for processed data

# Ensure directories exist
ensure_directories()
