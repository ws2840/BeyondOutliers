#!/usr/bin/env python3
"""
Choose the style factors: K of the 153 original JKP characteristics.

The style factors are ORIGINAL characteristics, used as JKP provides them - not principal
components, and not composites of several characteristics. PCA is used only to decide which
ones: it shows how many independent directions the 153 factor returns really have, and one
original characteristic is then taken to stand for each direction.

    1. Candidates   characteristics with data for at least --min-coverage of S&P 500
                    stock-months. A factor most stocks have no value for cannot describe them.
    2. PCA          on the covariance of JKP's daily factor returns (the public file from
                    jkpfactors.com), TRAINING YEARS ONLY (common.TRAIN_END), so the hidden
                    test years have no say in which factors the model uses.
    3. How many     read off the table: the share of factor-return variance the first K
                    components cover. It flattens quickly; K = 12 covers ~94%.
    4. Which ones   pivoted QR on the top-K loadings (market_data/global_factor_reduce.py's
                    pick_by_pca): the K original characteristics whose returns best span the
                    top-K components, one per direction, with no two picked for the same reason.

The exact picks are not unique - many of the 153 are near-substitutes (several betas, several
momentum windows), and a different sample picks a different member of the same family. Read the
list as "one representative per theme", not as the 12 best characteristics.

Inputs : data/jkp/factor_returns_daily.parquet, data/jkp/all.parquet (+ reduced.parquet, universe)
Outputs: data/jkp/style_factors.csv       the chosen factors, in pick order (exposures.py reads this)
         data/jkp/style_factors_pca.csv   the PCA table: variance covered by each component

Run (from repo root):
    python Python/riskmodel/factor_selection.py
    python Python/riskmodel/factor_selection.py --k 15
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_data.global_factor_common import RETURNS_PATH, load_meta   # noqa: E402
from market_data.global_factor_reduce import pick_by_pca, pca          # noqa: E402

try:
    from riskmodel.common import STYLE_PATH, STYLE_PCA_PATH, TRAIN_END, TRADING_DAYS
    from riskmodel.exposures import load_panel
except ImportError:
    from common import STYLE_PATH, STYLE_PCA_PATH, TRAIN_END, TRADING_DAYS
    from exposures import load_panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=12, help="number of style factors")
    p.add_argument("--min-coverage", type=float, default=0.90,
                   help="least share of S&P 500 stock-months a characteristic must have data for")
    p.add_argument("--train-end", default=str(TRAIN_END.date()), help="last date the choice may look at")
    a = p.parse_args()
    train_end = pd.Timestamp(a.train_end)
    meta = load_meta().set_index("factor")

    # 1. candidates: enough data among the stocks the model is for
    panel = load_panel(meta.index.tolist(), verbose=False)
    panel = panel[panel["in_sp500"] & (panel["eom"] <= train_end)]
    coverage = panel[meta.index].notna().mean()
    cand = coverage.index[coverage >= a.min_coverage].tolist()
    print(f"candidates: {len(cand)} of {len(meta)} characteristics have data for >= {a.min_coverage:.0%} of "
          f"S&P 500 stock-months; dropped: {', '.join(coverage.index[coverage < a.min_coverage])}")

    # 2. PCA on the training years
    R = pd.read_parquet(RETURNS_PATH)
    R.index = pd.to_datetime(R.index)
    R = R.loc[R.index <= train_end, cand].dropna()
    print(f"JKP daily factor returns: {len(R):,} days, {R.index.min().date()} -> {R.index.max().date()}")
    lam, _ = pca(np.cov(R.to_numpy(), rowvar=False))
    table = pd.DataFrame({"pc": np.arange(1, len(lam) + 1), "explained_pct": 100 * lam / lam.sum(),
                          "cumulative_pct": 100 * np.cumsum(lam) / lam.sum()})
    table.to_csv(STYLE_PCA_PATH, index=False)
    print("variance of the factor returns covered by the first K components: "
          + ", ".join(f"K={k}: {table['cumulative_pct'].iloc[k - 1]:.1f}%" for k in (5, 8, 10, 12, 15, 20) if k <= len(lam)))

    # 3-4. one original characteristic per component
    picked, _, share, main_pc = pick_by_pca(R, a.k)
    out = meta.loc[picked, ["cluster", "description", "direction"]].copy()
    out.insert(0, "rank", range(1, len(picked) + 1))
    out["closest_pc"] = main_pc
    out["coverage_sp500"] = coverage[picked].round(3)
    out["vol_ann"] = (R[picked].std() * np.sqrt(TRADING_DAYS)).round(4)
    out.index.name = "factor"
    out.to_csv(STYLE_PATH)
    print(f"\nthe {a.k} picked characteristics span {100 * share:.1f}% of the variance of all {len(cand)} factor returns "
          f"(the top {a.k} components themselves: {table['cumulative_pct'].iloc[a.k - 1]:.1f}%)\n")
    pd.set_option("display.width", 200)
    print(out.assign(description=out["description"].str.slice(0, 44)).to_string())
    print(f"\nsaved {STYLE_PATH} and {STYLE_PCA_PATH.name}")


if __name__ == "__main__":
    main()
