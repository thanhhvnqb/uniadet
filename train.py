"""Train with the selected softmax-first defaults; see uniadet.cli for flags."""
import sys
from uniadet.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
