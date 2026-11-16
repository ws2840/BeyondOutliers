#!/usr/bin/env python3
"""
Build the whole risk model with one command.

    python Python/riskmodel/pipeline.py                 # full rebuild (~15 min)
    python Python/riskmodel/pipeline.py --from coverage # resume from a step
    python Python/riskmodel/pipeline.py --validate      # also run the bias test
    python Python/riskmodel/pipeline.py --dry-run       # just print what would run

The steps must run in order; each consumes the previous one's output. See
data/riskmodel/README.md for what each file is.

    1 exposures       data/jkp/all.parquet          -> exposures.parquet
    2 factor_returns  + data/returns/1h_bar.parquet -> factor_returns_hourly, residuals_hourly,
                                                       factor_returns_halfA/B_hourly
    3 coverage        + the above                   -> exposures_covered, residuals_covered
    4 alpha_target    residuals_*                   -> alpha_target_1d.parquet
    5 bias_test       (validation, optional)        -> bias_test_hourly.csv

Prerequisites, produced by Python/market_data/ (see data/jkp/README.md):
    data/jkp/all.parquet          global_factor_download.py   (needs WRDS)
    data/jkp/reduced.parquet      global_factor_reduce.py     (for the permno->ticker map)
    data/returns/1h_bar.parquet   return.py                   (needs Databento)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

STEPS = [
    ("exposures",      ["exposures.py"],                            "153 characteristics -> 13 composites + sector (B)"),
    ("factor_returns", ["factor_returns.py", "--freq", "hourly",
                        "--min-abs-t", "0"],                        "one cross-sectional regression per bar (f, u)"),
    ("coverage",       ["coverage.py"],                             "extend B and residuals to 100% of the index"),
    ("alpha_target",   ["alpha_target.py"],                         "sum 7 bars -> 1-day forecast target"),
]
VALIDATE = ("bias_test", ["bias_test.py", "--freq", "hourly", "--every", "30"],
            "validate: predicted vs realised risk")


def check_inputs() -> list[str]:
    need = [ROOT / "data" / "jkp" / "all.parquet",
            ROOT / "data" / "jkp" / "reduced.parquet",
            ROOT / "data" / "returns" / "1h_bar.parquet"]
    return [str(p.relative_to(ROOT)) for p in need if not p.exists()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", default=None,
                   choices=[s[0] for s in STEPS], help="resume from this step")
    p.add_argument("--only", default=None, choices=[s[0] for s in STEPS])
    p.add_argument("--validate", action="store_true", help="also run the bias test at the end")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    missing = check_inputs()
    if missing and not a.dry_run:
        print("missing prerequisites — build these first (see data/jkp/README.md):")
        for m in missing:
            print(f"    {m}")
        return 1

    steps = list(STEPS)
    if a.only:
        steps = [s for s in steps if s[0] == a.only]
    elif a.start:
        steps = steps[[s[0] for s in steps].index(a.start):]
    if a.validate:
        steps.append(VALIDATE)

    print(f"risk model pipeline: {len(steps)} step(s)\n")
    t0 = time.time()
    for i, (name, cmd, what) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name}: {what}")
        full = [sys.executable, str(HERE / cmd[0])] + cmd[1:]
        if a.dry_run:
            print(f"        {' '.join(full)}\n")
            continue
        t = time.time()
        r = subprocess.run(full, cwd=ROOT)
        if r.returncode != 0:
            print(f"\nFAILED at step '{name}'. Fix it, then resume with:")
            print(f"    python Python/riskmodel/pipeline.py --from {name}")
            return r.returncode
        print(f"        done in {time.time() - t:.0f}s\n")

    if not a.dry_run:
        print(f"pipeline complete in {time.time() - t0:.0f}s")
        print("outputs in data/riskmodel/ — see its README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
