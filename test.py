"""Evaluate UniADet and save the paper's seven metrics to JSON and CSV."""

import sys
from uniadet.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    main()
