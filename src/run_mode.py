"""
run_mode.py - ENABLE DEMO MODE STRAIGHT FROM THE COMMAND LINE
=============================================================
Demo mode is controlled by the FL_DEMO environment variable, and `config.py` reads that
variable AT IMPORT TIME. It therefore cannot be enabled through argparse: by the time
`args = ap.parse_args()` runs, config has already been loaded with the real paths.

Solution: inspect sys.argv directly BEFORE importing config.

Usage - put this immediately above `import config`:

    from src.run_mode import enable_demo_mode
    enable_demo_mode()
    import config

Then declare the flag for argparse so it does not reject an unknown argument:

    ap.add_argument("--demo", action="store_true", help="...")

Advantage over setting the environment variable by hand: the flag only affects THAT
process and does not leak into subsequent commands.
"""
from __future__ import annotations

import os
import sys

DEMO_FLAG = "--demo"


def enable_demo_mode(argv: list[str] | None = None) -> bool:
    """If --demo is on the command line, set FL_DEMO before config is loaded."""
    if DEMO_FLAG in (argv if argv is not None else sys.argv):
        os.environ["FL_DEMO"] = "1"
        return True
    return False


def print_demo_banner(cfg) -> None:
    """Print a banner so viewers know this is the scaled-down demo mode."""
    if not getattr(cfg, "DEMO_MODE", False):
        return
    line = "=" * 74
    print(line)
    print("  DEMO MODE - SCALED DOWN TO ILLUSTRATE THE PIPELINE")
    print(line)
    print(f"  Data     : {os.path.basename(cfg.DATA_DIR)}/     "
          f"(the real data lives in data/ - UNTOUCHED)")
    print(f"  Results  : {os.path.basename(cfg.RESULTS_DIR)}/  "
          f"(the real results live in results/ - UNTOUCHED)")
    print(f"  Scale    : {cfg.DATASET_MAX_SAMPLES} emails | "
          f"{cfg.NUM_CLIENTS} clients | {cfg.NUM_ROUNDS} rounds | "
          f"seq {cfg.MAX_SEQ_LENGTH}")
    print("             (full run: 8000 emails | 10 clients | 10 rounds | seq 128)")
    print("  CAVEAT   : numbers produced in this mode only illustrate the pipeline;")
    print("             they have no scientific value.")
    print(line)
    print()


def require_demo_data(cfg) -> None:
    """Stop early with a clear message if the demo data has not been prepared yet."""
    if not getattr(cfg, "DEMO_MODE", False):
        return
    if os.path.isdir(cfg.CLIENT_DATA_DIR) and os.listdir(cfg.CLIENT_DATA_DIR):
        return
    print("=" * 74)
    print("  DEMO DATA NOT FOUND")
    print("=" * 74)
    print("  Run these two commands first (about 15 seconds each):\n")
    print("      python scripts/01_prepare_data.py --demo")
    print("      python scripts/02_make_poison.py --demo\n")
    print("=" * 74)
    sys.exit(1)
