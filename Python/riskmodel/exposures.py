#!/usr/bin/env python3
"""
Build the risk model's exposure matrix B.

Instead of using the 153 JKP characteristics as 153 separate factors (which is not
estimable on a ~457-stock S&P 500 cross-section, and whose covariance has a condition
number of ~4e5), this aggregates all 153 into the 13 clusters JKP defines. Each cluster
becomes one composite style factor, so every characteristic contributes but the factor
set stays well conditioned and interpretable. This is the descriptor -> style-factor
aggregation used by commercial risk models (Barra, Axioma).

Per month-end, for each of the 153 characteristics:
    winsorize at +-3 sd  ->  cap-weighted z-score (mean 0, sd 1)  ->  x JKP `direction`
Then per cluster: average the signed z-scores a stock actually has (missing ones are
skipped, not treated as 0), and re-z-score the composite.

DIRECTION. JKP's clusters group characteristics whose SIGNED long-short returns move
together, and most clusters mix signs: Size holds market_equity (-1) next to ami_126d (+1),
Value holds 14 positive and 4 negative. Averaging raw z-scores lets those cancel. Each
z-score is therefore multiplied by its `direction` from factor_meta.csv first, so a high
composite always means "more of what the cluster is named after".

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
    <13 cluster composites>

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
    from riskmodel.common import ROOT, INSTRUMENT_PATH, MEMBERSHIP_PATH, GICS_SECTORS, MIN_SECTOR, norm_ticker
except ImportError:
    from common import ROOT, INSTRUMENT_PATH, MEMBERSHIP_PATH, GICS_SECTORS, MIN_SECTOR, norm_ticker

OUT_PATH = ROOT / "data" / "riskmodel" / "exposures.parquet"


def zscore_month(X: pd.DataFrame, cap: pd.Series) -> pd.DataFrame:
    """Winsorize each column at +-3 sd, then cap-weighted z-score. NaNs stay NaN."""
    mu, sd = X.mean(), X.std()
    X = X.clip(mu - 3 * sd, mu + 3 * sd, axis=1)
    w = cap / cap.sum()
    wmu = X.mul(w, axis=0).sum() / X.notna().mul(w, axis=0).sum()
    return (X - wmu) / X.std().replace(0, np.nan)


def cluster_composites(panel: pd.DataFrame, clusters: dict[str, list[str]],
                       direction: pd.Series) -> pd.DataFrame:
    """One composite exposure per cluster per month: mean of the signed z-scores present, re-z-scored."""
    cols = [c for members in clusters.values() for c in members]
    sign = direction.reindex(cols).astype(float)
    out = []
    for eom, g in panel.groupby("eom", sort=True):
        cap = g["me"].astype(float)
        Z = zscore_month(g[cols].astype(float), cap) * sign      # orient every characteristic
        comp = pd.DataFrame(index=g.index)
        for name, members in clusters.items():
            comp[name] = Z[members].mean(axis=1, skipna=True)    # skip missing, don't zero-fill
        comp = zscore_month(comp, cap).fillna(0.0)               # re-standardize, then missing -> average stock
        comp.insert(0, "eom", eom)
        out.append(comp)
    return pd.concat(out)


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

    meta = load_meta()
    clusters = {c: sorted(g["factor"]) for c, g in meta.groupby("cluster")}
    direction = meta.set_index("factor")["direction"]
    print(f"{len(meta)} characteristics -> {len(clusters)} cluster composites")
    for c, cols in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        d = direction.reindex(cols)
        print(f"  {c:22s} {len(cols):3d}   ({int((d > 0).sum())} positive, {int((d < 0).sum())} negative direction)")

    id_cols = ["permno", "eom", "me", "gics", "ff49"]      # all.parquet has no ticker column
    src = pd.read_parquet(ALL_PATH, columns=list(dict.fromkeys(id_cols + meta["factor"].tolist())))
    src["eom"] = pd.to_datetime(src["eom"])
    src = src[src["permno"].notna() & src["me"].notna() & (src["me"] > 0)].reset_index(drop=True)
    src["permno"] = src["permno"].astype(int)

    # tickers are attached during the reduce step; bring them over on (permno, eom)
    tick = pd.read_parquet(REDUCED_PATH, columns=["permno", "eom", "ticker", "ticker_cls"])
    tick["eom"] = pd.to_datetime(tick["eom"])
    tick = tick[tick["permno"].notna()]
    tick["permno"] = tick["permno"].astype(int)
    src = src.merge(tick.drop_duplicates(["permno", "eom"]), on=["permno", "eom"], how="left")
    print(f"ticker attached for {src['ticker'].notna().mean():.1%} of rows")
    print(f"\npanel: {len(src):,} rows, {src['eom'].nunique()} month-ends, "
          f"{src['eom'].min().date()} -> {src['eom'].max().date()}")

    src["key"] = sp500_key(src)
    src["in_sp500"] = src["key"] != ""
    # two share classes of one company can both match the same member: keep the larger
    dup = src["in_sp500"] & src.duplicated(["eom", "key"], keep=False)
    if dup.any():
        best = src[dup].sort_values("me", ascending=False).drop_duplicates(["eom", "key"]).index
        lose = src.index[dup].difference(best)
        print(f"{len(lose):,} rows are a second share class of an index member "
              f"({', '.join(sorted(src.loc[lose, 'key'].unique()))}); the larger class is kept")
        src.loc[lose, ["key", "in_sp500"]] = ["", False]
    not_member = ~src["in_sp500"]
    src.loc[not_member, "key"] = norm_ticker(src.loc[not_member, "ticker_cls"].fillna("")).to_numpy()

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

    comp = cluster_composites(src, clusters, direction)
    out = pd.DataFrame({"permno": src["permno"], "eom": src["eom"], "ticker": src["ticker"],
                        "ticker_cls": src["ticker_cls"], "key": src["key"], "me": src["me"].astype(float),
                        "in_sp500": src["in_sp500"], "sector": src["sector"],
                        "sector_src": src["sector_src"], "sector_model": src["sector_model"]})
    out = pd.concat([out, comp.drop(columns="eom")], axis=1)

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, compression="zstd")
    print(f"\nsaved {path}: {out.shape[0]:,} rows x {out.shape[1]} cols")
    return out, clusters


if __name__ == "__main__":
    main()
