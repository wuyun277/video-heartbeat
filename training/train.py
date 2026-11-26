"""Backward-compatible command entry: python -m training.train."""
from pathlib import Path
import sys

# Also retain direct invocation: python training/train.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.cli import main

if __name__ == "__main__":
    main()
