#!/usr/bin/env python3
"""
Factor-based performance attribution, laid out like Axioma "Statistical Models" (2024), Figure 1.

Rows:  Portfolio / Benchmark / Active
         Specific Return
         Factor Contribution
           Style   (each JKP factor)
           Market Intercept
           Sectors (each GICS sector, optional)
Cols:  Contribution      cumulative over the period; monthly pieces are Carino-linked so that
                         Active = compounded Portfolio - compounded Benchmark and every row adds up
       Avg Active Exp.   time-average of x_t = X_t' w_a,t (style: in sd units; sectors: in % weight)
       Realized Risk     annualized std of that row's monthly contribution series

Model each month t (exposures at month-end t, returns over t+1):
    r_{t+1} = 1*f_mkt + S_t f_sector + X_t f_style + u     (WLS, weights sqrt(me))
    sector returns constrained to cap-weighted sum 0, so the intercept is the market
    X_t = JKP characteristics, winsorized +-3 sd, cap-weighted z-score, missing -> 0
Contribution of factor k in month t+1:  x_{k,t} * f_{k,t+1};  specific: w_a,t' u_{t+1}
These add up exactly to the active return w_a,t' r_{t+1}.

Inputs : data/jkp/all.parquet
         --weights CSV with columns eom, permno, weight [, bmk_weight]   (one block per rebalance)
         default (demo): equal-weight universe vs cap-weighted universe, rebalanced monthly
Outputs: data/jkp/attribution/attribution_table.csv / .xlsx, monthly_contributions.csv

Run (from repo root):
    python Python/riskmodel/factor_attribution.py --factors-from reduced
    python Python/riskmodel/factor_attribution.py --factors-from reduced --weights backtest_weights.csv \
        --start 2021-01-31 --end 2024-02-29
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market_data.global_factor_common import ALL_PATH, JKP_DIR, PICKED_PATH, load_meta  # noqa: E402

ANN = 12
GICS_SECTORS = {10: "Energy", 15: "Materials", 20: "Industrials", 25: "Consumer Discretionary",
                30: "Consumer Staples", 35: "Health Care", 40: "Financials", 45: "Information Technology",
                50: "Communication Services", 55: "Utilities", 60: "Real Estate"}


def standardize(X, cap):
    mu, sd = X.mean(), X.std()
    X = X.clip(mu - 3 * sd, mu + 3 * sd, axis=1)
    w = cap / cap.sum()
    wmu = X.mul(w, axis=0).sum() / X.notna().mul(w, axis=0).sum()
    return ((X - wmu) / X.std().replace(0, np.nan)).fillna(0.0)


def sector_of(df):
    s = (pd.to_numeric(df["gics"], errors="coerce") // 1_000_000).map(GICS_SECTORS)
    return s.fillna("Unclassified")


def weights_for(eom, g, wt):
    """Portfolio and benchmark weights on the stocks in g (renormalized to stocks with returns)."""
    cap = g["me"] / g["me"].sum()
    if wt is None:
        w = pd.Series(1 / len(g), index=g.index)
        b = cap
    else:
        blk = wt[wt["eom"] == eom].set_index("permno")
        if blk.empty:
            return None, None
        w = g["permno"].map(blk["weight"]).fillna(0.0)
        b = g["permno"].map(blk["bmk_weight"]).fillna(0.0) if "bmk_weight" in blk else cap
    if w.sum() <= 0 or b.sum() <= 0:
        return None, None
    return w / w.sum(), b / b.sum()


def carino(rp, rb, contrib):
    """Link monthly arithmetic contributions so they sum to compounded rp - compounded bp."""
    def k(a, b):
        return np.where(np.isclose(a, b), 1 / (1 + a), (np.log1p(a) - np.log1p(b)) / (a - b))
    Rp, Rb = np.prod(1 + rp) - 1, np.prod(1 + rb) - 1
    kt = k(rp, rb)
    K = k(np.array([Rp]), np.array([Rb]))[0]
    return contrib.mul(kt / K, axis=0).sum(), Rp, Rb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factors-from", default="reduced", choices=["all", "reduced"])
    p.add_argument("--weights", default=None)
    p.add_argument("--start", default=None, help="first rebalance month-end (default: first available)")
    p.add_argument("--end", default=None, help="last rebalance month-end")
    p.add_argument("--no-sectors", action="store_true")
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--out", default=str(JKP_DIR / "attribution"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = load_meta().set_index("factor")
    styles = (pd.read_csv(PICKED_PATH)["factor"].tolist() if a.factors_from == "reduced"
              else meta.index.tolist())
    cols = ["permno", "eom", "me", "gics", "ret_exc_lead1m"]
    panel = pd.read_parquet(ALL_PATH, columns=list(dict.fromkeys(cols + styles)))
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel = panel[panel["permno"].notna() & (panel["me"] > 0) & panel["ret_exc_lead1m"].notna()]
    panel["permno"] = panel["permno"].astype(int)
    if a.start:
        panel = panel[panel["eom"] >= a.start]
    if a.end:
        panel = panel[panel["eom"] <= a.end]

    wt = None
    if a.weights:
        wt = pd.read_csv(a.weights, parse_dates=["eom"])
        wt["permno"] = wt["permno"].astype(int)

    rows, sector_names = [], sorted(set(GICS_SECTORS.values()) | {"Unclassified"})
    for eom, g in panel.groupby("eom", sort=True):
        g = g.reset_index(drop=True)
        w, b = weights_for(eom, g, wt)
        if w is None:
            continue
        wa = (w - b).values
        X = standardize(g[styles].astype(float), g["me"])
        parts, names = [np.ones((len(g), 1))], ["Market Intercept"]
        if not a.no_sectors:
            sec = sector_of(g)
            present = [s for s in sector_names if (sec == s).any()]
            S = np.column_stack([(sec == s).values.astype(float) for s in present])
            c = (S * g["me"].values[:, None]).sum(0)
            R = np.vstack([np.eye(len(present) - 1), -c[:-1] / c[-1]])  # cap-weighted sum of sector returns = 0
            parts.append(S @ R)
        parts.append(X.values)
        A = np.column_stack(parts)
        if len(g) < 3 * A.shape[1]:
            continue
        sw = np.sqrt(g["me"].values)
        Aw, yw = A * sw[:, None], g["ret_exc_lead1m"].values * sw
        P = np.eye(A.shape[1]) * a.ridge
        P[0, 0] = 0
        coef = np.linalg.solve(Aw.T @ Aw + P, Aw.T @ yw)
        u = g["ret_exc_lead1m"].values - A @ coef

        f_mkt = coef[0]
        nsec = 0 if a.no_sectors else len(present) - 1
        f_sec = (R @ coef[1:1 + nsec]) if nsec else np.array([])
        f_sty = coef[1 + nsec:]
        rec = {"date": eom + pd.offsets.MonthEnd(1),
               "Portfolio": w.values @ g["ret_exc_lead1m"].values,
               "Benchmark": b.values @ g["ret_exc_lead1m"].values,
               "Specific Return": wa @ u,
               "c|Market Intercept": wa.sum() * f_mkt, "x|Market Intercept": wa.sum()}
        if nsec:
            xs = S.T @ wa
            for s, xk, fk in zip(present, xs, f_sec):
                rec[f"c|{s}"], rec[f"x|{s}"] = xk * fk, xk
        xs = X.values.T @ wa
        for s, xk, fk in zip(styles, xs, f_sty):
            rec[f"c|{s}"], rec[f"x|{s}"] = xk * fk, xk
        rows.append(rec)

    m = pd.DataFrame(rows).set_index("date").fillna(0.0)
    m["Active"] = m["Portfolio"] - m["Benchmark"]
    c_cols = [c for c in m.columns if c.startswith("c|")]
    check = (m["Specific Return"] + m[c_cols].sum(axis=1) - m["Active"]).abs().max()
    print(f"{len(m)} months, {m.index.min().date()} -> {m.index.max().date()}; "
          f"max |active - sum of pieces| = {check:.2e}")
    m.to_csv(out / "monthly_contributions.csv")

    contrib = m[["Specific Return"] + c_cols].rename(columns=lambda c: c.replace("c|", ""))
    sec_present = [s for s in sector_names if s in contrib.columns]
    contrib["Style"] = contrib[styles].sum(axis=1)
    contrib["Sectors"] = contrib[sec_present].sum(axis=1) if sec_present else 0.0
    contrib["Factor Contribution"] = contrib[["Style", "Market Intercept", "Sectors"]].sum(axis=1)
    linked, Rp, Rb = carino(m["Portfolio"].values, m["Benchmark"].values, contrib)
    risk = contrib.std() * np.sqrt(ANN)
    avgx = m[[c for c in m.columns if c.startswith("x|")]].mean().rename(lambda c: c.replace("x|", ""))

    def row(name, level, label=None):
        is_sec = name in sec_present or name == "Market Intercept"
        x = avgx.get(name, np.nan)
        return {"Source of Return": "  " * level + (label or name),
                "Contribution": linked.get(name, np.nan),
                "Avg Active Exposure": f"{100 * x:.2f}%" if is_sec and pd.notna(x) else (f"{x:.4f}" if pd.notna(x) else ""),
                "Realized Risk": risk.get(name, np.nan)}

    tbl = [
        {"Source of Return": "Portfolio", "Contribution": Rp, "Avg Active Exposure": "",
         "Realized Risk": m["Portfolio"].std() * np.sqrt(ANN)},
        {"Source of Return": "Benchmark", "Contribution": Rb, "Avg Active Exposure": "",
         "Realized Risk": m["Benchmark"].std() * np.sqrt(ANN)},
        {"Source of Return": "  Active", "Contribution": Rp - Rb, "Avg Active Exposure": "",
         "Realized Risk": m["Active"].std() * np.sqrt(ANN)},
        row("Specific Return", 2), row("Factor Contribution", 2), row("Style", 3),
    ]
    tbl += [row(s, 4, f"{s} ({meta.loc[s, 'description']})") for s in sorted(styles)]
    tbl += [row("Market Intercept", 3)]
    if sec_present:
        tbl += [row("Sectors", 3)] + [row(s, 4) for s in sec_present]
    tbl = pd.DataFrame(tbl)
    tbl.to_csv(out / "attribution_table.csv", index=False)

    rank = pd.DataFrame({"contribution": linked[styles], "avg_active_exposure": avgx[styles],
                         "realized_risk": risk[styles]})
    rank = rank.join(meta[["cluster", "description"]])
    rank = rank.reindex(rank["contribution"].abs().sort_values(ascending=False).index)
    rank.insert(0, "rank", range(1, len(rank) + 1))
    rank.to_csv(out / "style_ranking.csv")

    pct = lambda v: f"{100 * v:.2f}%" if pd.notna(v) else ""
    show = tbl.copy()
    show["Contribution"] = show["Contribution"].map(pct)
    show["Realized Risk"] = show["Realized Risk"].map(pct)
    with pd.ExcelWriter(out / "attribution.xlsx") as xw:
        show.to_excel(xw, sheet_name="attribution_table", index=False)
        rank.to_excel(xw, sheet_name="style_ranking")
        m.to_excel(xw, sheet_name="monthly")
    label = "DEMO equal-weight vs cap-weight" if wt is None else a.weights
    print(f"\nFactor-based performance attribution ({label}, excess returns)\n")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
