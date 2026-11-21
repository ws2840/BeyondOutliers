"""Download daily Nasdaq BBOs and sample hourly spreads for a complete trading month."""
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import databento as db
import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __package__:
    from .universe import get_universe
else:
    from universe import get_universe

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "spread"
DATASET = "XNAS.ITCH"
SCHEMA = "bbo-1m"
UNDEF_PRICE = 2**63 - 1  # DBN price for a side with no quote
NY = "America/New_York"
MINUTE = timedelta(minutes=1)
RAW_META = b"beyondoutliers_bbo"
DEPTH = ["bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00"]


def download_spread_data(start_date="2025-01-01", end_date="2025-02-01", *, output_dir=OUTPUT_DIR, client=None) -> list[Path]:
    """Cache Nasdaq minute BBOs for each session's universe in [start_date, end_date).
    Reuse raw files or existing Databento orders. Submit one batch for each day still missing.
    Return the paths to raw/YYYY-MM-DD.parquet in trading-date order.
    """
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start >= end:
        raise ValueError("end_date is exclusive and must follow start_date")
    calendar = xcals.get_calendar("XNYS", start=start - timedelta(days=7), end=end + timedelta(days=7))
    sessions = calendar.schedule.loc[start:end - timedelta(days=1)]
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    missing = {}
    for day, session in sessions.iterrows():
        path = raw_dir / f"{day.date()}.parquet"
        paths.append(path)
        if path.exists():
            continue
        tickers = get_universe(day.date())
        if not tickers:
            raise ValueError(f"Empty universe for {day.date()}")
        # Right-end minute labels run from 09:31 through the close; the API end is exclusive.
        missing[path] = dict(dataset=DATASET, schema=SCHEMA, stype_in="raw_symbol", symbols=tickers,
                             start=(session.open + MINUTE).isoformat(), end=(session.close + MINUTE).isoformat())
    if missing:
        if client is None:
            client = db.Historical()
        _SpreadJobs(client).download(missing, raw_dir)
    return paths


def build_hour_bar_spread(month="2025-01", *, output_dir=OUTPUT_DIR) -> Path:
    """Sample the last same-session BBO for each ticker at each hourly decision, 10:00–15:00 New York.
    Read local raw files and return the path to monthly/YYYY-MM.parquet. Early closes stop at 12:00.
    """
    period = pd.Period(month, freq="M")
    calendar = xcals.get_calendar("XNYS", start=period.start_time, end=period.end_time.normalize())
    daily_bars = []
    for day, session in calendar.schedule.iterrows():
        raw_path = Path(output_dir) / "raw" / f"{day.date()}.parquet"
        tickers, book = _read_session_book(raw_path, day, session)
        daily_bars.append(_hourly_spreads(book, tickers, session))
    path = Path(output_dir) / "monthly" / f"{period}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as folder:
        tmp = Path(folder) / "hourly.parquet"
        pd.concat(daily_bars, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(path)
    return path


class _SpreadJobs:
    """Submit or recover daily BBO orders, then retrieve them by their saved Databento IDs."""

    def __init__(self, client):
        self._batch = client.batch
        self._remote_ids = None

    def download(self, missing, raw_dir):
        # Submit all missing days before waiting on the first one. The journal survives an interrupted run.
        with closing(sqlite3.connect(raw_dir.parent / "jobs.sqlite3", timeout=30)) as journal:
            journal.execute("""CREATE TABLE IF NOT EXISTS jobs (request_key TEXT PRIMARY KEY,
                               request TEXT NOT NULL, job_id TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
            jobs = [(path, request, self._job_for(path, request, journal)) for path, request in missing.items()]

        for path, request, job_id in jobs:
            for attempt in range(3):  # Repeat retrieval and writing with the same paid ID.
                try:
                    self._save_day(path, request, job_id)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)

    def _job_for(self, path, request, journal):
        """Recover a matching order locally or from Databento. Reserve a new request before submitting it."""
        request_json = json.dumps(request, sort_keys=True)
        key = hashlib.sha256(request_json.encode()).hexdigest()
        saved = journal.execute("SELECT job_id FROM jobs WHERE request_key=?", (key,)).fetchone()
        if saved and saved[0]:
            return saved[0]

        if self._remote_ids is None:
            self._remote_ids = self._existing_orders(request)
        job_id = self._remote_ids.get(request_json)
        with journal:
            reservation = journal.execute("INSERT OR IGNORE INTO jobs(request_key,request,job_id) VALUES (?,?,?)",
                                          (key, request_json, job_id))
        if not job_id and not reservation.rowcount:  # A prior submit may have timed out after reaching Databento.
            raise RuntimeError(f"Unconfirmed batch for {path.stem}. Check Download Center, then rerun.")
        if not job_id:
            for attempt in range(3):
                try:
                    job = self._batch.submit_job(**request, encoding="dbn", compression="zstd", split_duration="none")
                    job_id = job["id"]
                    break
                except db.BentoClientError as error:
                    if error.http_status == 429 and attempt < 2:
                        time.sleep(int(error.headers.get("Retry-After", 60)))
                        continue
                    if error.http_status in {400, 401, 402, 403, 413, 422, 429}:  # order explicitly rejected
                        with journal:
                            journal.execute("DELETE FROM jobs WHERE request_key=? AND job_id IS NULL", (key,))
                    raise
        with journal:
            journal.execute("UPDATE jobs SET job_id=? WHERE request_key=?", (job_id, key))
        return job_id

    def _existing_orders(self, request):
        """Index prior BBO orders by the same symbol list, time interval and feed as a daily request."""
        remote_ids = {}
        formats = dict(dataset=DATASET, schema=SCHEMA, encoding="dbn", compression="zstd", stype_out="instrument_id")
        for entry in self._batch.list_jobs(states=["queued", "processing", "done", "expired"]):
            job = self._batch.get_job_details(entry["id"])
            if any(job[field] != value for field, value in formats.items()):
                continue
            if job.get("limit") or job.get("delivery", "download") != "download":
                continue
            fields = {field: job[field] for field in request}
            symbols = fields["symbols"]
            if isinstance(symbols, str):
                symbols = symbols.split(",")
            fields["symbols"] = sorted(symbols)
            for field in ("start", "end"):
                fields[field] = pd.Timestamp(job[field]).isoformat()
            identity = json.dumps(fields, sort_keys=True)
            # An expired match prevents a second charge. Prefer any order that can still be downloaded.
            if job["state"] != "expired" or identity not in remote_ids:
                remote_ids[identity] = job["id"]
        return remote_ids

    def _save_day(self, path, request, job_id):
        """Wait for one ID and save its original DBN fields and request metadata as a daily Parquet."""
        state = self._batch.get_job_details(job_id)["state"]
        while state in {"queued", "processing"}:
            time.sleep(5)
            state = self._batch.get_job_details(job_id)["state"]
        if state != "done":
            raise RuntimeError(f"Batch {job_id} is {state}")
        with TemporaryDirectory(dir=path.parent) as folder:
            files = self._batch.download(job_id=job_id, output_dir=folder)
            dbn_files = [part for part in files if part.name.endswith(".dbn.zst")]
            parts = []
            for part in dbn_files:
                with part.open("rb") as stream:
                    store = db.DBNStore.from_bytes(stream)
                    parts.append(store.to_df(price_type="fixed", pretty_ts=False))
            raw = pd.concat(parts).reset_index()  # keep ts_recv, the DBN index
            info = dict(request=request, job_id=job_id, rows=len(raw), price_scale_usd=1e-9,
                        undef_price=UNDEF_PRICE, version=1, compressed_dbn_bytes=sum(p.stat().st_size for p in dbn_files),
                        timestamps="UTC Unix ns; ts_recv = interval end, ts_event = last trade")
            table = pa.Table.from_pandas(raw, preserve_index=False)
            table = table.replace_schema_metadata({**table.schema.metadata, RAW_META: json.dumps(info).encode()})
            tmp = Path(folder) / "raw.parquet"
            pq.write_table(table, tmp, compression="zstd")
            tmp.replace(path)


def _read_session_book(raw_path, day, session):
    """Read the requested universe and its regular-session BBOs, with prices in dollars per share."""
    info = json.loads(pq.read_schema(raw_path).metadata[RAW_META])
    request = info["request"]
    tickers = get_universe(day.date())
    same_feed = request["dataset"] == DATASET and request["schema"] == SCHEMA
    same_open = pd.Timestamp(request["start"]) == session.open + MINUTE
    same_close = pd.Timestamp(request["end"]) == session.close + MINUTE
    if request["symbols"] != tickers or not (same_feed and same_open and same_close):
        raise ValueError(f"Raw BBO file does not match this session: {raw_path}")

    book = pd.read_parquet(raw_path, columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00", *DEPTH])
    book = book.rename(columns={"symbol": "ticker", "bid_px_00": "bid", "ask_px_00": "ask"})
    # ts_recv = 10:00 labels [09:59, 10:00), so it matches the 10:00 decision directly.
    book["source_ts"] = pd.to_datetime(book.pop("ts_recv"), unit="ns", utc=True).dt.tz_convert(NY)
    in_session = book["source_ts"].between(session.open, session.close, inclusive="right")
    book = book.loc[in_session].copy()
    prices = book[["bid", "ask"]]
    book[["bid", "ask"]] = prices.mask(prices == UNDEF_PRICE).astype(float) / 1e9  # dollars per share
    book[DEPTH] = book[DEPTH].astype("UInt32")
    return tickers, book


def _hourly_spreads(book, tickers, session):
    """Match the last same-session snapshot to each decision and compute midpoint, spreads and quote status."""
    first_hour = session.open.tz_convert(NY).ceil("h")
    hours = pd.date_range(first_hour, session.close.tz_convert(NY), freq="h", inclusive="left")
    hours = hours[(hours.hour >= 10) & (hours.hour <= 15)].as_unit("ns")
    grid = pd.MultiIndex.from_product([hours, tickers], names=["ts", "ticker"]).to_frame(index=False)
    book["ticker"] = book["ticker"].astype(grid["ticker"].dtype)
    # An empty minute takes the last snapshot (price and size) for this ticker today, or NaN if none.
    bars = pd.merge_asof(grid, book.sort_values("source_ts"), left_on="ts", right_on="source_ts",
                         by="ticker", direction="backward")
    bars["age_seconds"] = (bars["ts"] - bars["source_ts"]).dt.total_seconds()
    bars["carried"] = bars["age_seconds"] > 0

    bid, ask = bars["bid"], bars["ask"]
    has_bid, has_ask = bid.gt(0), ask.gt(0)
    both_sides = has_bid & has_ask
    bars["mid"] = ((bid + ask) / 2).where(both_sides)
    bars["spread"] = (ask - bid).round(9).where(both_sides & (ask >= bid))
    bars["half_spread"] = bars["spread"] / 2  # one-way cost from mid, in dollars per share
    bars["spread_bps"] = 10_000 * bars["spread"] / bars["mid"]
    bars["half_spread_bps"] = bars["spread_bps"] / 2

    wide = (100 * bars["spread"] - bars["mid"]).round(10) > 0  # strictly above 100 bps (1% of mid)
    quality = {
        "missing": bars["source_ts"].isna(),
        "no_valid_side": ~has_bid & ~has_ask,
        "one_sided": has_bid != has_ask,
        "crossed": ask < bid,
        "locked": ask == bid,
        "wide": wide,
    }
    bars["status"] = np.select(list(quality.values()), list(quality), default="ok")  # First match wins.
    return bars


if __name__ == "__main__":
    monthly_files = []
    for month in pd.period_range("2025-01", "2025-12", freq="M"):
        download_spread_data(month.start_time.date(), (month + 1).start_time.date())
        monthly_files.append(build_hour_bar_spread(str(month)))

    yearly = OUTPUT_DIR / "yearly" / "2025.parquet"
    yearly.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=yearly.parent) as folder:
        tmp = Path(folder) / yearly.name
        rows = pd.concat([pd.read_parquet(path) for path in monthly_files], ignore_index=True)
        rows.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(yearly)
    print(yearly)
