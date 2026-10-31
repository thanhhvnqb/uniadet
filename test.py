"""Evaluate using saved checkpoint settings, including softmax order and Gaussian sigma."""

import sys
from uniadet.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
