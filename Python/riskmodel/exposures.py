#!/usr/bin/env python3
"""
Build the risk model's exposure matrix B.

STYLE FACTORS. Each style factor is one ORIGINAL JKP characteristic, used as JKP provides it.
Which ones is decided once by factor_selection.py (PCA on the training years) and read here
from data/jkp/style_factors.csv. Per month-end, across the S&P 500 stocks, each chosen
characteristic is

    winsorized at +-3 sd  ->  standardized to cap-weighted mean 0, sd 1  ->  x JKP `direction`

and a stock with no value gets 0, the average stock. This is the standardization commercial
risk models use (Barra): cap-weighted centring makes the index itself style-neutral, so the
market factor carries the market.

DIRECTION. JKP signs every characteristic so that a high value is the side its factor return
is long (market_equity is -1: the size factor is long SMALL stocks). Multiplying by it keeps
each exposure pointing the same way as JKP's published factor of the same name.

IDENTITY. Each row is resolved to `key`, the normalized ticker of the S&P 500 member it is
(BRK + class B -> BRKB), which is what returns, residuals and the optimizer's weights are
keyed on. Where two share classes of one company both match a member (LEN / LEN.B), the
larger one is kept.

SECTORS. `sector` is the GICS sector and `sector_src` says where it came from:
    jkp          JKP `gics` at that month-end (point in time)
    jkp_carried  the stock's own most recent earlier JKP `gics` (point in time)
    current_list data/universe/sp500_instrument.csv, by ticker. NOT point in time: it is
                 today's classification of today's members, applied to the past. JKP has no
                 `gics` at all for ~100 index members, so this cannot be avoided without a
                 historical GICS source; where both exist they disagree on 8% of 2018 rows,
                 about 1% in 2023 and under 0.5% after.
    none         unknown -> "Other"
`sector_model` is the
sector FACTOR the stock loads on: its own sector if at least MIN_SECTOR estimation stocks
share it that month, otherwise the pooled "Other" factor, otherwise none (""). Deciding this
once per month-end, here, is what keeps the regression (factor_returns.py) and the served
exposures (model.py) on the same definition.

Output: data/riskmodel/exposures.parquet
    permno, eom, ticker, ticker_cls, key, me, in_sp500, sector, sector_src, sector_model,
    <one column per style factor, named after its JKP characteristic>

Run (from repo root):
    python Python/riskmodel/exposures.py
    python Python/riskmodel/exposures.py --universe all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market_data.global_factor_common import ALL_PATH, REDUCED_PATH, load_meta  # noqa: E402

try:
    from riskmodel.common import (ROOT, INSTRUMENT_PATH, MEMBERSHIP_PATH, STYLE_PATH, GICS_SECTORS, MIN_SECTOR,
                                  norm_ticker)
except ImportError:
    from common import ROOT, INSTRUMENT_PATH, MEMBERSHIP_PATH, STYLE_PATH, GICS_SECTORS, MIN_SECTOR, norm_ticker

OUT_PATH = ROOT / "data" / "riskmodel" / "exposures.parquet"


def zscore_month(X: pd.DataFrame, cap: pd.Series) -> pd.DataFrame:
    """Winsorize each column at +-3 sd, then cap-weighted z-score. NaNs stay NaN."""
    mu, sd = X.mean(), X.std()
    X = X.clip(mu - 3 * sd, mu + 3 * sd, axis=1)
    w = cap / cap.sum()
    wmu = X.mul(w, axis=0).sum() / X.notna().mul(w, axis=0).sum()
    return (X - wmu) / X.std().replace(0, np.nan)


def style_exposures(panel: pd.DataFrame, styles: list[str], direction: pd.Series) -> pd.DataFrame:
    """Each chosen characteristic, standardized within the month and signed. Missing -> 0."""
    sign = direction.reindex(styles).astype(float)
    out = []
    for _, g in panel.groupby("eom", sort=True):
        out.append((zscore_month(g[styles].astype(float), g["me"].astype(float)) * sign).fillna(0.0))
    return pd.concat(out).reindex(panel.index)


def load_panel(characteristics: list[str], verbose: bool = True) -> pd.DataFrame:
    """JKP stock-months with the given characteristics, each row resolved to the index member it is.

    Adds ticker, ticker_cls, key and in_sp500 (see IDENTITY). Rows keep JKP's `gics` and `me`.
    """
    id_cols = ["permno", "eom", "me", "gics", "ff49"]      # all.parquet has no ticker column
    src = pd.read_parquet(ALL_PATH, columns=list(dict.fromkeys(id_cols + list(characteristics))))
    src["eom"] = pd.to_datetime(src["eom"])
    src = src[src["permno"].notna() & src["me"].notna() & (src["me"] > 0)].reset_index(drop=True)
    src["permno"] = src["permno"].astype(int)

    # tickers are attached during the reduce step; bring them over on (permno, eom)
    tick = pd.read_parquet(REDUCED_PATH, columns=["permno", "eom", "ticker", "ticker_cls"])
    tick["eom"] = pd.to_datetime(tick["eom"])
    tick = tick[tick["permno"].notna()]
    tick["permno"] = tick["permno"].astype(int)
    src = src.merge(tick.drop_duplicates(["permno", "eom"]), on=["permno", "eom"], how="left")

    src["key"] = sp500_key(src)
    src["in_sp500"] = src["key"] != ""
    # two share classes of one company can both match the same member: keep the larger
    dup = src["in_sp500"] & src.duplicated(["eom", "key"], keep=False)
    if dup.any():
        best = src[dup].sort_values("me", ascending=False).drop_duplicates(["eom", "key"]).index
        lose = src.index[dup].difference(best)
        if verbose:
            print(f"{len(lose):,} rows are a second share class of an index member "
                  f"({', '.join(sorted(src.loc[lose, 'key'].unique()))}); the larger class is kept")
        src.loc[lose, ["key", "in_sp500"]] = ["", False]
    not_member = ~src["in_sp500"]
    src.loc[not_member, "key"] = norm_ticker(src.loc[not_member, "ticker_cls"].fillna("")).to_numpy()
    if verbose:
        print(f"ticker attached for {src['ticker'].notna().mean():.1%} of rows")
        print(f"panel: {len(src):,} rows, {src['eom'].nunique()} month-ends, "
              f"{src['eom'].min().date()} -> {src['eom'].max().date()}")
    return src


def attach_sector(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """GICS sector and where it came from (see SECTORS in the module docstring).

    Point-in-time sources are used first: JKP's `gics` for the month, then the same stock's
    last known earlier `gics`. Only what is still missing falls back to today's member list.
    """
    sector = (pd.to_numeric(df["gics"], errors="coerce") // 1_000_000).map(GICS_SECTORS)
    src = pd.Series(np.where(sector.notna(), "jkp", None), index=df.index, dtype=object)
    order = df["eom"].sort_values(kind="stable").index           # carry FORWARD in time only
    carried = sector.loc[order].groupby(df["permno"].loc[order]).ffill().reindex(df.index)
    src = src.where(sector.notna() | carried.isna(), "jkp_carried")
    sector = carried
    if INSTRUMENT_PATH.exists():
        inst = pd.read_csv(INSTRUMENT_PATH)
        imap = dict(zip(norm_ticker(inst["Symbol"]), inst["GICS Sector"]))
        for col in ("key", "ticker_cls", "ticker"):
            if col in df.columns:
                fill = norm_ticker(df[col].fillna("")).map(imap).set_axis(df.index)
                src = src.where(sector.notna() | fill.isna(), "current_list")
                sector = sector.fillna(fill)
    return sector.fillna("Other"), src.fillna("none")


def sp500_key(df: pd.DataFrame) -> pd.Series:
    """The S&P 500 member each row is at eom, as a normalized ticker; "" if it is not one.

    A row is matched under its bare CRSP ticker. Ticker + share class (BRK + B -> BRKB) is tried
    only against members that are themselves written with a class (BRK.B, BF.B). Without that
    restriction the concatenation invents matches: Mastercard is ticker MA, class A, and "MAA"
    is Mid-America Apartment; Pluralsight is PS, class A, and "PSA" is Public Storage.
    """
    out = pd.Series("", index=df.index, dtype=object)
    if not MEMBERSHIP_PATH.exists() or "ticker" not in df.columns:
        return out
    m = pd.read_csv(MEMBERSHIP_PATH, parse_dates=["start_date", "end_date"])
    m["key"] = norm_ticker(m["ticker"])
    m["has_class"] = m["ticker"].astype(str).str.contains(r"[.\-/]", regex=True)
    cand = [pd.DataFrame({"key": norm_ticker(df[c].fillna("")).to_numpy(), "eom": df["eom"].to_numpy(),
                          "row": np.arange(len(df)), "with_class": c == "ticker_cls"})
            for c in ("ticker", "ticker_cls") if c in df.columns]
    x = pd.concat(cand)
    x = x[x["key"] != ""]
    j = x.merge(m[["key", "start_date", "end_date", "has_class"]], on="key", how="inner")
    j = j[(j["start_date"] <= j["eom"]) & (j["end_date"].isna() | (j["end_date"] > j["eom"]))]
    j = j[j["with_class"] == j["has_class"]]             # bare ticker <-> plain member, ticker+class <-> classed member
    j = j.sort_values(["row", "with_class"]).drop_duplicates("row", keep="first")
    out.iloc[j["row"].to_numpy()] = j["key"].to_numpy()
    return out


def model_sector(df: pd.DataFrame) -> pd.Series:
    """Which sector factor each row loads on (see SECTORS in the module docstring)."""
    est = df[df["in_sp500"]]
    n = est.groupby(["eom", "sector"]).size().rename("n").reset_index()
    own = df[["eom", "sector"]].merge(n, on=["eom", "sector"], how="left")["n"].fillna(0).to_numpy() >= MIN_SECTOR
    pooled = pd.Series(np.where(own, df["sector"], "Other"), index=df.index)
    n_other = pooled[df["in_sp500"]].groupby(df.loc[df["in_sp500"], "eom"]).apply(lambda s: (s == "Other").sum())
    other_ok = df["eom"].map(n_other).fillna(0).to_numpy() >= MIN_SECTOR
    return pooled.where((pooled != "Other") | other_ok, "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="sp500", choices=["sp500", "all"],
                   help="cross-section the z-scores are computed on. The model serves an S&P 500 "
                        "portfolio, so standardizing on all 4,800 US stocks would leave the S&P 500 "
                        "systematically off-centre (-1.3 sd on Size), making Size collinear with the "
                        "market intercept.")
    p.add_argument("--out", default=str(OUT_PATH))
    a = p.parse_args()

    if not STYLE_PATH.exists():
        sys.exit(f"{STYLE_PATH} not found: run `python Python/riskmodel/factor_selection.py` first")
    chosen = pd.read_csv(STYLE_PATH)
    styles = chosen["factor"].tolist()
    direction = load_meta().set_index("factor")["direction"]
    print(f"{len(styles)} style factors from {STYLE_PATH.name}:")
    for f, d, desc in zip(chosen["factor"], chosen["direction"], chosen["description"]):
        print(f"  {f:18s} {'+' if d > 0 else '-'}  {desc}")

    src = load_panel(styles)
    src["sector"], src["sector_src"] = attach_sector(src)
    if a.universe == "sp500":
        src = src[src["in_sp500"]].reset_index(drop=True)
        print(f"standardizing within the S&P 500: {len(src):,} rows, "
              f"{int(src.groupby('eom').size().median())} stocks per month")
    src["sector_model"] = model_sector(src)
    share = src.loc[src["in_sp500"], "sector_src"].value_counts(normalize=True)
    print("sector source (S&P 500 rows): " + ", ".join(f"{k} {v:.1%}" for k, v in share.items()))
    last = src[src["in_sp500"] & (src["eom"] == src["eom"].max())]
    print(f"sector factors at {src['eom'].max().date()}: "
          + ", ".join(f"{s or '(none)'} {n}" for s, n in last["sector_model"].value_counts().sort_index().items()))
    missing = src.loc[src["in_sp500"], styles].isna().mean()
    print("share of S&P 500 stock-months with no value (set to 0): "
          + (", ".join(f"{f} {v:.1%}" for f, v in missing[missing > 0.005].items()) or "none above 0.5%"))

    X = style_exposures(src, styles, direction)
    out = pd.DataFrame({"permno": src["permno"], "eom": src["eom"], "ticker": src["ticker"],
                        "ticker_cls": src["ticker_cls"], "key": src["key"], "me": src["me"].astype(float),
                        "in_sp500": src["in_sp500"], "sector": src["sector"],
                        "sector_src": src["sector_src"], "sector_model": src["sector_model"]})
    out = pd.concat([out, X], axis=1)

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, compression="zstd")
    print(f"\nsaved {path}: {out.shape[0]:,} rows x {out.shape[1]} cols")
    return out, styles


if __name__ == "__main__":
    main()
