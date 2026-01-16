#!/usr/bin/env python3
"""
Generate the interactive paper-section hub HTML for the project.

Output:
  results/interactive/paper_hub.html
"""

import sys
from pathlib import Path

# Allow importing sibling script as a module
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from paper_hub import generate_paper_hub  # type: ignore


def main() -> int:
    generate_paper_hub()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

