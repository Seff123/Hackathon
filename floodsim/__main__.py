"""FloodSim command line. Subcommands are filled in as phases land."""
from __future__ import annotations

import sys


def main(argv=None):
    from .cli import main as _main
    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
