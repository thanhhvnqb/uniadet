import sys
from uniadet.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    main()
