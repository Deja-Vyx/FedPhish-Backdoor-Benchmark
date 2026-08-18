"""
run_logger.py - CAPTURE EVERYTHING A RUN PRINTS INTO LOG FILES
==============================================================
Every demo run produces TWO files in results/logs/:

    demoN_YYYYmmdd_HHMMSS_full.log     - verbatim copy of everything printed to the
                                         screen, including Ray and Flower output. Use it
                                         when you need to trace an error.
    demoN_YYYYmmdd_HHMMSS.log          - a de-noised version keeping only the meaningful
                                         lines, meant for READING AND ANALYSIS.

Usage inside a demo script:

    from src.run_logger import start_logging, stop_logging
    session = start_logging("demo1")     # at the very top of main()
    ...
    stop_logging(session)                # at the very bottom of main()

The filter KEEPS: scenario headers, [Round N] lines, client information, summary tables,
real warnings and every traceback. It DROPS: Flower DEPRECATED notices, Ray internal
logs and Hugging Face token warnings.
"""
from __future__ import annotations

import io
import logging
import os
import platform
import re
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------
# A line matching any of these patterns is removed from the filtered log.
NOISE_PATTERNS = [
    r"DEPRECATED FEATURE",
    r"This is a deprecated feature",
    r"It will be removed",
    r"entirely in future versions",
    r"Using `start_simulation\(\)` is deprecated",
    r"\$ flwr (new|run)",
    r"Instead, use the `flwr run` CLI",
    r"^\(raylet\)",
    r"ray/_private",
    r"Ray deduplicates logs",
    r"ray-observability",
    r"unauthenticated requests to the HF Hub",
    r"Optimize your simulation with Flower VCE",
    r"flower\.ai/docs",
    r"Started a local Ray instance",
    r"^\s*$",                       # blank lines emitted by Ray
]
_RE_NOISE = re.compile("|".join(NOISE_PATTERNS))

# A line matching these is ALWAYS kept, even if it also matches a noise pattern above.
KEEP_PATTERNS = [
    r"\[Round \d+\]",
    r"\[Client \d+\]",
    r">>>",
    r"^#{5,}",
    r"^={5,}",
    r"^-{5,}",
    r"\[STALE\]|\[CACHED\]|\[SKIP\]|\[ERROR\]",
    r"Traceback|Error|Exception|error:",
    r"attack=|defense=",
    r"ASR|CA=|MCC|accuracy",
    r"Total time|Saved|finished",
]
_RE_KEEP = re.compile("|".join(KEEP_PATTERNS), re.IGNORECASE)

_RE_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Prefix Ray attaches to every line printed by a worker - dropped for readability.
_RE_PREFIX = re.compile(r"^\((?:ClientAppActor|raylet|pid=)[^)]*\)\s?")
# Ray deduplicates logs and appends this suffix - not needed in the filtered version.
_RE_SUFFIX = re.compile(r"\s*\[repeated \d+x across cluster\]\s*$")


def _strip_ansi(s: str) -> str:
    return _RE_ANSI.sub("", s)


def _clean_line(line: str) -> str:
    """Remove Ray's prefix/suffix so the log line reads cleanly."""
    return _RE_SUFFIX.sub("", _RE_PREFIX.sub("", line)).rstrip()


def _short_path(p: str) -> str:
    """Use a relative path when it is short, otherwise keep the absolute one."""
    try:
        r = os.path.relpath(p)
        return r if not r.startswith("..") else p
    except Exception:
        return p


def _is_noise(line: str) -> bool:
    d = _strip_ansi(line)
    if _RE_KEEP.search(d):
        return False
    return bool(_RE_NOISE.search(d))


# ---------------------------------------------------------------------------
class _Tee(io.TextIOBase):
    """Duplicate an output stream: still shown on screen, also written to two files."""

    def __init__(self, original, full_file, filtered_file):
        self.original = original
        self.full = full_file
        self.filtered = filtered_file
        self._buffer = ""            # holds the part that has no newline yet

    def write(self, s):
        self.original.write(s)
        self.full.write(_strip_ansi(s))
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not _is_noise(line):
                cleaned = _clean_line(_strip_ansi(line))
                if cleaned.strip():
                    self.filtered.write(cleaned + "\n")
        return len(s)

    def flush(self):
        for f in (self.original, self.full, self.filtered):
            try:
                f.flush()
            except Exception:
                pass

    def isatty(self):
        return getattr(self.original, "isatty", lambda: False)()


class LogSession:
    """Holds the state of one logging session."""

    def __init__(self, name, full_path, filtered_path, f1, f2, t0):
        self.name = name
        self.full_path = full_path
        self.filtered_path = filtered_path
        self._f1, self._f2 = f1, f2
        self._t0 = t0
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._handler = None


# ---------------------------------------------------------------------------
def start_logging(demo_name: str, directory: str | None = None) -> LogSession:
    """Start mirroring everything printed to the screen into files. Returns the session."""
    if directory is None:
        # Follow config.RESULTS_DIR so that --demo runs log into results_demo/ and never
        # write anything into the real results/ directory.
        try:
            import config as _cfg
            directory = os.path.join(_cfg.RESULTS_DIR, "logs")
        except Exception:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            directory = os.path.join(root, "results", "logs")
    os.makedirs(directory, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p_full = os.path.join(directory, f"{demo_name}_{stamp}_full.log")
    p_filtered = os.path.join(directory, f"{demo_name}_{stamp}.log")

    f1 = open(p_full, "w", encoding="utf-8", buffering=1)
    f2 = open(p_filtered, "w", encoding="utf-8", buffering=1)

    t0 = time.time()
    session = LogSession(demo_name, p_full, p_filtered, f1, f2, t0)

    # --- File header: the run context, so the log stays self-explanatory later ---
    try:
        import torch
        device = (f"CUDA {torch.version.cuda} - "
                  f"{torch.cuda.get_device_name(0)}"
                  if torch.cuda.is_available() else "CPU")
        torch_version = torch.__version__
    except Exception:
        device, torch_version = "unknown", "unknown"

    try:
        import config as _cfg
        cfg_block = (
            f"  Clients          : {_cfg.NUM_CLIENTS}\n"
            f"  Rounds           : {_cfg.NUM_ROUNDS}\n"
            f"  Model            : {_cfg.CLASSIFIER_MODEL_NAME}\n"
            f"  Dataset size     : {_cfg.DATASET_MAX_SAMPLES}\n"
            f"  Malicious ratio  : {_cfg.DEFAULT_MALICIOUS_RATIO}\n"
            f"  Poison ratio     : {_cfg.POISON_RATIO}\n"
            f"  Random seed      : {_cfg.RANDOM_SEED}\n"
            f"  Schema version   : {_cfg.RESULTS_SCHEMA_VERSION}\n"
        )
    except Exception:
        cfg_block = "  (could not read config)\n"

    header = (
        "=" * 74 + "\n"
        f"RUN LOG - {demo_name.upper()}\n"
        + "=" * 74 + "\n"
        f"  Started at       : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"  Command          : {' '.join(sys.argv)}\n"
        f"  Machine          : {platform.node()} - {platform.platform()}\n"
        f"  Python           : {platform.python_version()}\n"
        f"  PyTorch          : {torch_version}\n"
        f"  Compute device   : {device}\n"
        + cfg_block
        + "=" * 74 + "\n\n"
    )
    for f in (f1, f2):
        f.write(header)

    # --- Redirect stdout / stderr ---
    sys.stdout = _Tee(session._stdout, f1, f2)
    sys.stderr = _Tee(session._stderr, f1, f2)

    # --- Also capture library logging (Flower uses the logging module) ---
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(h)
    session._handler = h

    print("[LOG] Writing to:")
    print(f"      {_short_path(p_filtered)}   (filtered - meant for reading)")
    print(f"      {_short_path(p_full)}   (verbatim)\n")
    return session


def stop_logging(session: LogSession | None) -> None:
    """End the logging session, close the files and print their paths."""
    if session is None:
        return
    seconds = time.time() - session._t0
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    footer = (
        "\n" + "=" * 74 + "\n"
        f"  Finished at      : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"  Total time       : "
        f"{f'{hours}h ' if hours else ''}{minutes}m {secs}s\n"
        + "=" * 74 + "\n"
    )
    print(footer)

    try:
        logging.getLogger().removeHandler(session._handler)
    except Exception:
        pass
    sys.stdout, sys.stderr = session._stdout, session._stderr
    for f in (session._f1, session._f2):
        try:
            f.flush()
            f.close()
        except Exception:
            pass

    print("[LOG] Saved:")
    print(f"      {_short_path(session.filtered_path)}   (filtered - meant for reading)")
    print(f"      {_short_path(session.full_path)}   (verbatim)")
