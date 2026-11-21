"""Download daily Nasdaq BBOs and sample spreads at exact decision hours."""
import hashlib
import json
import shutil
import sqlite3
import time
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import databento as db
import exchange_calendars as xcals
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


def download_spread_data(start_date="2025-01-01", end_date="2026-01-01", *, output_dir=OUTPUT_DIR, client=None) -> list[Path]:
    """Cache each session's minute BBOs in YYYY-MM-DD.parquet for [start_date, end_date)."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start >= end:
        raise ValueError("end_date is exclusive and must follow start_date")
    calendar = xcals.get_calendar("XNYS", start=start - timedelta(days=7), end=end + timedelta(days=7))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths, missing = [], {}
    for day, session in calendar.schedule.loc[start:end - timedelta(days=1)].iterrows():
        path, request = _daily_raw(root, day, session)
        target = root / path.name
        if path != target and not target.exists():  # Promote an older raw/ cache without placing a new order.
            with TemporaryDirectory(dir=root) as folder:
                tmp = Path(folder) / path.name
                shutil.copyfile(path, tmp)
                tmp.replace(target)
            path = target
        if not path.exists():
            missing[path] = request
        paths.append(path)
    if missing:
        if client is None:
            client = db.Historical()
        _SpreadJobs(client).download(missing, root)
    return paths


def build_hour_bar_spread(year=2025, *, month=None, output_dir=OUTPUT_DIR) -> Path:
    """Build 1h_bar.parquet from exact minute BBOs. Replace only the requested year or month."""
    selection = str(month if month is not None else year)
    period = pd.Period(selection, freq="Y" if len(selection) == 4 else "M")
    calendar = xcals.get_calendar("XNYS", start=period.start_time, end=period.end_time.normalize())
    root = Path(output_dir)
    daily = []
    for day, session in calendar.schedule.iterrows():
        raw, request = _daily_raw(root, day, session)
        daily.append(_hourly_spreads(raw, session, request["symbols"]))
    bars = pd.concat(daily, ignore_index=True)

    path = root / "1h_bar.parquet"
    # Serialize range replacements so two concurrent monthly runs preserve each other's rows.
    with closing(sqlite3.connect(root / "jobs.sqlite3", timeout=30)) as journal, journal:
        journal.execute("BEGIN IMMEDIATE")
        if path.exists():
            previous = pd.read_parquet(path)
            start, end = period.start_time.tz_localize(NY), (period + 1).start_time.tz_localize(NY)
            keep = (previous["ts"] < start) | (previous["ts"] >= end)
            bars = pd.concat([previous.loc[keep], bars], ignore_index=True)
        bars = bars.sort_values(["ts", "ticker"], ignore_index=True)
        with TemporaryDirectory(dir=root) as folder:
            tmp = Path(folder) / path.name
            bars.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)
    return path


def _daily_raw(root, day, session):
    """Locate a daily raw file and verify its feed, trading window and point-in-time universe."""
    tickers = get_universe(day.date())
    if not tickers:
        raise ValueError(f"Empty universe for {day.date()}")
    # The first regular minute ends at 09:31. The exclusive API end includes the close label.
    request = dict(dataset=DATASET, schema=SCHEMA, stype_in="raw_symbol", symbols=tickers,
                   start=(session.open + MINUTE).isoformat(), end=(session.close + MINUTE).isoformat())
    target = root / f"{day.date()}.parquet"
    legacy = root / "raw" / target.name
    for path in (target, legacy):
        if not path.is_file():
            continue
        metadata = pq.read_schema(path).metadata or {}
        if RAW_META not in metadata:  # Some 2018 pilot files at the root are already hourly bars.
            continue
        stored = json.loads(metadata[RAW_META])["request"]
        if (any(stored[field] != request[field] for field in ("dataset", "schema", "stype_in", "symbols"))
                or any(pd.Timestamp(stored[field]) != pd.Timestamp(request[field]) for field in ("start", "end"))):
            raise ValueError(f"Raw BBO file does not match this session: {path}")
        return path, request
    if target.exists() or legacy.exists():
        raise ValueError(f"No raw BBO file for {day.date()}; an existing Parquet is not a minute quote cache")
    return target, request


class _SpreadJobs:
    """Submit or recover daily BBO orders, then retrieve them by their saved Databento IDs."""

    def __init__(self, client):
        self._batch = client.batch
        self._remote_ids = None

    def download(self, missing, root):
        # Submit all missing days before waiting on the first one. The journal survives an interrupted run.
        with closing(sqlite3.connect(root / "jobs.sqlite3", timeout=30)) as journal:
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
            info = dict(request=request, rows=len(raw), price_scale_usd=1e-9,
                        undef_price=UNDEF_PRICE, version=1, compressed_dbn_bytes=sum(p.stat().st_size for p in dbn_files),
                        timestamps="UTC Unix ns; ts_recv = interval end, ts_event = last trade")
            table = pa.Table.from_pandas(raw, preserve_index=False)
            table = table.replace_schema_metadata({**table.schema.metadata, RAW_META: json.dumps(info).encode()})
            tmp = Path(folder) / "raw.parquet"
            pq.write_table(table, tmp, compression="zstd")
            tmp.replace(path)


def _hourly_spreads(path, session, tickers):
    """Match the exact BBO at 10:00–15:00 New York (10:00–12:00 on an early close)."""
    first = session.open.tz_convert(NY).ceil("h")
    hours = pd.date_range(first, session.close.tz_convert(NY), freq="h", inclusive="left")
    hours = hours[(hours.hour >= 10) & (hours.hour <= 15)]
    # A 10:00 ts_recv is the right edge of [09:59, 10:00). A missing minute contributes no row.
    bars = pd.read_parquet(path, columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"],
                           filters=[("ts_recv", "in", [hour.value for hour in hours])])
    bars = bars.rename(columns={"symbol": "ticker", "bid_px_00": "bid", "ask_px_00": "ask"})
    bars = bars.loc[bars["ticker"].isin(tickers)].copy()
    bars["ticker"] = bars["ticker"].astype(object)
    bars["ts"] = pd.to_datetime(bars.pop("ts_recv"), unit="ns", utc=True).dt.tz_convert(NY)
    prices = bars[["bid", "ask"]]
    bars[["bid", "ask"]] = prices.mask(prices == UNDEF_PRICE).astype(float) / 1e9
    bid, ask = bars["bid"], bars["ask"]
    valid = bid.gt(0) & ask.gt(0) & ask.ge(bid)
    bars["half_spread"] = (ask - bid).round(9).where(valid) / 2  # one-way cost, dollars per share
    return bars[["ts", "ticker", "bid", "ask", "half_spread"]].drop_duplicates(["ts", "ticker"], keep="last")


if __name__ == "__main__":
    download_spread_data()
    print(build_hour_bar_spread())
