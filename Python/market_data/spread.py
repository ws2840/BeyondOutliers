"""Sample Nasdaq minute BBOs at the portfolio's hourly decisions.

Run `python Python/market_data/spread.py` for May 2018, or supply
`--start 2018-05-01 --end 2018-05-02` for one day. Dates are New York
session dates in [start, end). The Databento SDK uses DATABENTO_API_KEY.
One local parquet file per session contains one row per (ts, ticker).
Bid, ask, mid, spread and half_spread are USD/share. spread_bps and
half_spread_bps scale their dollar equivalents by 10,000 / mid.
"""

import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import exchange_calendars as xcals
import numpy as np
import pandas as pd

if __package__:
    from .universe import get_universe
else:  # python Python/market_data/spread.py
    from universe import get_universe

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "spread"
DATASET = "XNAS.ITCH"
SCHEMA = "bbo-1m"
TZ = "America/New_York"
MINUTE = timedelta(minutes=1)
DEFAULT_START = date(2018, 5, 1)
DEFAULT_END = date(2018, 6, 1)


def decision_times(market_open, market_close):
    """Return exact NY hours from 10:00 to 15:00, strictly before session close."""
    opened = market_open.tz_convert(TZ)
    hours = pd.date_range(opened.ceil("h"), market_close.tz_convert(TZ), freq="h", inclusive="left")
    return hours[(hours.hour >= 10) & (hours.hour <= 15)].as_unit("ns")


def spread_snapshot(quotes, tickers, times, market_open):
    """Match right-labeled BBO records to decisions in one regular session.

    source_ts is the end of the BBO snapshot interval, which can contain
    a carried-forward book. age_seconds measures snapshot age, not the age
    of the underlying exchange quote. Preserve both sides of the same row.
    """
    grid = pd.MultiIndex.from_product([times, tickers], names=["ts", "ticker"]).to_frame(index=False)
    if quotes.empty:
        grid["source_ts"] = pd.Series(pd.NaT, dtype=grid["ts"].dtype, index=grid.index)
        grid[["bid", "ask"]] = np.nan
    else:
        # DBNStore.to_df indexes BBO by ts_recv. ts_event is the LAST TRADE.
        if "ts_recv" not in quotes.columns:
            if quotes.index.name != "ts_recv":
                raise ValueError("bbo-1m needs ts_recv as a column or named index")
            quotes = quotes.reset_index()
        book = quotes[["ts_recv", "symbol", "bid_px_00", "ask_px_00"]].rename(columns={
            "ts_recv": "source_ts", "symbol": "ticker", "bid_px_00": "bid", "ask_px_00": "ask",
        })
        book["source_ts"] = book["source_ts"].dt.tz_convert(TZ).astype(grid["ts"].dtype)
        # A BBO ending at the open describes the preceding pre-market minute.
        book = book.loc[book["ticker"].isin(tickers) & (book["source_ts"] > market_open)
                        & (book["source_ts"] <= times[-1])].copy()
        book["ticker"] = book["ticker"].astype(grid["ticker"].dtype)
        if book.duplicated(["source_ts", "ticker"]).any():
            raise ValueError("duplicate BBO records for a ticker and minute")
        grid = pd.merge_asof(grid, book.sort_values("source_ts"), left_on="ts", right_on="source_ts",
                             by="ticker", direction="backward", allow_exact_matches=True)

    grid["age_seconds"] = (grid["ts"] - grid["source_ts"]).dt.total_seconds()
    grid["carried"] = grid["age_seconds"].gt(0)
    bid, ask = grid["bid"], grid["ask"]
    # An undefined DBN price is INT64_MAX in units of 1e-9 USD.
    max_price = np.iinfo(np.int64).max / 1e9
    good_bid = (bid > 0) & (bid < max_price)
    good_ask = (ask > 0) & (ask < max_price)
    valid_prices = good_bid & good_ask
    grid["mid"] = ((bid + ask) / 2).where(valid_prices)
    # DBN prices have nine decimal places. Remove subtraction noise before
    # applying the strict 100-bps threshold (e.g. 9.95 bid / 10.05 ask).
    grid["spread"] = (ask - bid).round(9).where(valid_prices & (ask >= bid))
    grid["half_spread"] = grid["spread"] / 2
    grid["spread_bps"] = 10_000 * grid["spread"] / grid["mid"]
    grid["half_spread_bps"] = grid["spread_bps"] / 2
    # spread > 1% of mid. A midpoint can be half a DBN price unit.
    wide = (100 * grid["spread"] - grid["mid"]).round(10) > 0
    grid["status"] = np.select(
        [grid["source_ts"].isna(), ~good_bid & ~good_ask, ~valid_prices,
         ask < bid, ask == bid, wide],
        ["missing", "no_valid_side", "one_sided", "crossed", "locked", "wide"], default="ok",
    )
    return grid


def _check_day(frame, times, tickers):
    expected = pd.MultiIndex.from_product([times, tickers], names=["ts", "ticker"])
    actual = pd.MultiIndex.from_frame(frame[["ts", "ticker"]])
    if not actual.equals(expected) or not {"source_ts", "bid", "ask", "mid", "spread", "half_spread",
                                           "spread_bps", "half_spread_bps", "status",
                                           "age_seconds", "carried"}.issubset(frame.columns):
        raise ValueError("spread file has an unexpected schema or incomplete session grid")


def _daily_requests(start_date, end_date):
    start = pd.Timestamp(date.fromisoformat(str(start_date)))
    end = pd.Timestamp(date.fromisoformat(str(end_date)))
    if end <= start:
        raise ValueError("end_date must be later than start_date (exclusive end)")
    # Padding supports even a request containing only an exchange holiday.
    calendar = xcals.get_calendar("XNYS", start=start - timedelta(days=7), end=end + timedelta(days=7))
    sessions = calendar.schedule.loc[start:end - timedelta(days=1)]
    for session, schedule in sessions.iterrows():
        day = session.date()
        times = decision_times(schedule["open"], schedule["close"])
        tickers = get_universe(day)
        if not tickers:
            raise ValueError(f"empty project universe for {day}")
        request = dict(dataset=DATASET, schema=SCHEMA, symbols=tickers, stype_in="raw_symbol",
                       start=schedule["open"] + MINUTE, end=times[-1] + MINUTE)
        yield day, schedule["open"], times, request


def _historical_client():
    import databento as db
    return db.Historical()


def spread_size(start_date=DEFAULT_START, end_date=DEFAULT_END):
    """Offline upper bounds assuming one 80-byte BBO record per stock per requested minute."""
    rows = []
    for day, _, times, request in _daily_requests(start_date, end_date):
        stocks = len(request["symbols"])
        minutes = (request["end"] - request["start"]) // MINUTE
        maximum = stocks * minutes
        rows.append({"date": day, "stocks": stocks, "decisions": len(times), "stock_decisions": stocks * len(times),
                     "minutes": minutes, "max_records": maximum, "max_record_bytes": 80 * maximum})
    return pd.DataFrame(rows, columns=["date", "stocks", "decisions", "stock_decisions", "minutes",
                                        "max_records", "max_record_bytes"])


def estimate_spreads(start_date=DEFAULT_START, end_date=DEFAULT_END, *, client=None):
    """Query Databento metadata for the same per-day requests; prices are in USD."""
    estimates = []
    for day, _, _, request in _daily_requests(start_date, end_date):
        if client is None:
            client = _historical_client()
        meta = client.metadata
        estimates.append({"date": day, "records": meta.get_record_count(**request),
                          "billable_bytes": meta.get_billable_size(**request), "cost_usd": meta.get_cost(**request)})
    return pd.DataFrame(estimates, columns=["date", "records", "billable_bytes", "cost_usd"])


def build_spreads(start_date=DEFAULT_START, end_date=DEFAULT_END, *, output_dir=OUTPUT_DIR, client=None):
    """Write and return daily file paths for the half-open range [start_date, end_date)."""
    paths = []
    output_dir = Path(output_dir)
    for day, opened, times, request in _daily_requests(start_date, end_date):
        output_dir.mkdir(parents=True, exist_ok=True)
        tickers = request["symbols"]
        path = output_dir / f"{day}.parquet"
        if path.exists():
            try:
                _check_day(pd.read_parquet(path), times, tickers)
            except (ValueError, KeyError) as exc:
                warnings.warn(f"rebuilding invalid day file {path}: {exc}")
            else:
                paths.append(path)
                print(f"{day}: cached {len(tickers) * len(times):,} rows, {path.stat().st_size:,} bytes, {path}")
                continue

        if client is None:
            client = _historical_client()
        raw = client.timeseries.get_range(**request)
        quotes = raw.to_df(price_type="float", pretty_ts=True, map_symbols=True)
        frame = spread_snapshot(quotes, tickers, times, opened)
        with NamedTemporaryFile(dir=output_dir, prefix=f".{day}-", suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
        try:
            frame.to_parquet(temp, index=False, compression="zstd")
            _check_day(pd.read_parquet(temp), times, tickers)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        print(f"{day}: wrote {len(frame):,} rows, {path.stat().st_size:,} bytes, {path}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START, help="first session date (inclusive)")
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END, help="exclusive end date")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--size-only", action="store_true", help="offline upper bounds; no API key or network needed")
    mode.add_argument("--estimate", action="store_true", help="query Databento metadata without downloading quotes")
    args = parser.parse_args()
    if args.size_only:
        sizes = spread_size(args.start, args.end)
        print("Offline upper bound: up to one 80-byte uncompressed DBN record per stock per requested minute.")
        print(sizes.to_string(index=False))
        print("Total:", sizes[["stock_decisions", "max_records", "max_record_bytes"]].sum().to_dict())
    elif args.estimate:
        estimates = estimate_spreads(args.start, args.end)
        print(estimates.to_string(index=False))
        print("Total:", estimates[["records", "billable_bytes", "cost_usd"]].sum().to_dict())
    else:
        build_spreads(args.start, args.end)
