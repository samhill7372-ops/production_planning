#!/usr/bin/env python
"""
CLI script for training models.

Usage:
    python scripts/train_model.py yield-2024
    python scripts/train_model.py yield-2025
    python scripts/train_model.py --all
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.train import train_single_model, train_all_models


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_model.py <model_name>")
        print("       python scripts/train_model.py --all")
        print("\nAvailable models:")
        print("  - yield-2024")
        print("  - yield-2025")
        print("  - yield-2year")
        print("  - yield-5year")
        print("  - yield-6month")
        print("  - yield-multi")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--all" or arg == "-a":
        train_all_models()
    else:
        train_single_model(arg)


if __name__ == "__main__":
    main()
