"""
timing.py - MEASURE EXPERIMENT RUNTIME
=======================================
Kept separate from experiment.py so it can be unit-tested without torch or a GPU.

Used to:
  - Know how long each scenario takes (feeds the "computational cost" section of a report).
  - Estimate the remaining time while a long batch of scenarios is running.
  - Compare the cost of the different defences (FLTrust is more expensive because the
    server also trains on the root set; Krum costs O(n^2) distances; FoolsGold keeps
    history, and so on).
"""

import time
import numpy as np
import pandas as pd


def fmt_duration(seconds: float) -> str:
    """Format seconds in a readable way: 75 -> '1m15s', 3725 -> '1h02m'."""
    s = int(round(max(0.0, float(seconds))))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class RunTimer:
    """Collect the runtime of several scenarios within one demo."""

    def __init__(self, total_runs: int = None, label: str = ""):
        self.rows = []
        self.total_runs = total_runs
        self.label = label
        self.t0 = time.perf_counter()

    def add(self, meta: dict, quiet: bool = False):
        self.rows.append({
            "run": meta.get("run"),
            "attack": meta.get("attack"),
            "defense": meta.get("defense"),
            "num_rounds": meta.get("num_rounds"),
            "duration_seconds": meta.get("duration_seconds", 0.0),
            "duration": meta.get("duration_readable"),
            "seconds_per_round": meta.get("seconds_per_round"),
            "device": meta.get("device"),
        })
        if not quiet and self.total_runs and len(self.rows) < self.total_runs:
            done = len(self.rows)
            avg = float(np.mean([r["duration_seconds"] for r in self.rows]))
            remain = avg * (self.total_runs - done)
            print(f"    ->  Finished {done}/{self.total_runs} scenarios | "
                  f"estimated {fmt_duration(remain)} remaining")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def eta(self) -> float:
        """Estimate the remaining seconds from the average of the finished scenarios."""
        if not self.rows or not self.total_runs:
            return 0.0
        avg = float(np.mean([r["duration_seconds"] for r in self.rows]))
        return max(0.0, avg * (self.total_runs - len(self.rows)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def report(self) -> pd.DataFrame:
        df = self.to_frame()
        print("\n" + "=" * 74)
        print(f"RUNTIME{(' - ' + self.label) if self.label else ''}")
        print("=" * 74)
        if len(df):
            cols = [c for c in ["run", "attack", "defense", "num_rounds",
                                "duration", "seconds_per_round", "device"] if c in df]
            print(df[cols].to_string(index=False))
            print(f"\n  Total    : {fmt_duration(self.elapsed)} for {len(df)} scenarios")
            print(f"  Average  : {fmt_duration(df['duration_seconds'].mean())}/scenario")
            slow = df.loc[df["duration_seconds"].idxmax()]
            print(f"  Slowest  : {slow['run']} ({slow['duration']})")
        else:
            print("  (no scenario recorded yet)")
        return df
