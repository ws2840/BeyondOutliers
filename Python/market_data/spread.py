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
NY = "America/New_York"
MINUTE = timedelta(minutes=1)
RAW_META = b"beyondoutliers_bbo"
DEPTH = ["bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00"]


def download_spread_data(start_date="2025-01-01", end_date="2025-02-01", *, output_dir=OUTPUT_DIR, client=None):
    """Cache all regular-session minute BBOs for each day's universe. End date is exclusive."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start >= end:
        raise ValueError("end_date is exclusive and must follow start_date")
    calendar = xcals.get_calendar("XNYS", start=start - timedelta(days=7), end=end + timedelta(days=7))
    sessions = calendar.schedule.loc[start:end - timedelta(days=1)]
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths, missing = [], []
    for day, session in sessions.iterrows():
        path = raw_dir / f"{day.date()}.parquet"
        paths.append(path)
        if path.exists():
            continue
        # Minute labels mark the interval end, so include the close.
        request = dict(dataset="XNAS.ITCH", schema="bbo-1m", stype_in="raw_symbol",
                       symbols=get_universe(day.date()), start=(session.open + MINUTE).isoformat(),
                       end=(session.close + MINUTE).isoformat())
        if not request["symbols"]:
            raise ValueError(f"Empty universe for {day.date()}")
        missing.append((path, request))
    if not missing:
        return paths

    batch = (client if client is not None else db.Historical()).batch
    downloads = []
    remote_ids = None
    with closing(sqlite3.connect(raw_dir.parent / "jobs.sqlite3", timeout=30)) as journal:
        journal.execute("""CREATE TABLE IF NOT EXISTS jobs (request_key TEXT PRIMARY KEY,
                           request TEXT NOT NULL, job_id TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        for path, request in missing:
            encoded = json.dumps(request, sort_keys=True)
            key = hashlib.sha256(encoded.encode()).hexdigest()
            local = journal.execute("SELECT job_id FROM jobs WHERE request_key=?", (key,)).fetchone()
            if local and local[0]:
                downloads.append((path, request, local[0]))
                continue
            # Also reuse orders submitted on another machine. The list API returns only summaries.
            if remote_ids is None:
                remote_ids = {}
                for entry in batch.list_jobs(states=["queued", "processing", "done", "expired"]):
                    job = batch.get_job_details(entry["id"])
                    if (job["dataset"], job["schema"], job["encoding"], job["compression"]) != ("XNAS.ITCH", "bbo-1m", "dbn", "zstd"):
                        continue
                    if job["stype_out"] != "instrument_id" or job.get("limit") or job.get("delivery", "download") != "download":
                        continue
                    fields = {field: job[field] for field in request}
                    names = fields["symbols"].split(",") if isinstance(fields["symbols"], str) else fields["symbols"]
                    fields.update(symbols=sorted(names), start=pd.Timestamp(job["start"]).isoformat(),
                                  end=pd.Timestamp(job["end"]).isoformat())
                    identity = json.dumps(fields, sort_keys=True)
                    if job["state"] != "expired" or identity not in remote_ids:
                        remote_ids[identity] = job["id"]

            # Commit the claim before the paid POST, including when two processes run together.
            job_id = remote_ids.get(encoded)
            with journal:
                new = journal.execute("INSERT OR IGNORE INTO jobs(request_key,request,job_id) VALUES (?,?,?)",
                                      (key, encoded, job_id)).rowcount
            if not job_id and not new:
                raise RuntimeError(f"Unconfirmed batch for {path.stem}. Check Download Center, then rerun.")
            if not job_id:
                try:
                    job_id = batch.submit_job(**request, encoding="dbn", compression="zstd", split_duration="none")["id"]
                except Exception as error:
                    if getattr(error, "http_status", None) in {400, 401, 402, 403, 413, 422, 429}:
                        with journal:
                            journal.execute("DELETE FROM jobs WHERE request_key=? AND job_id IS NULL", (key,))
                    raise
            with journal:
                journal.execute("UPDATE jobs SET job_id=? WHERE request_key=?", (job_id, key))
            downloads.append((path, request, job_id))

    for path, request, job_id in downloads:
        for attempt in range(3):  # Only poll and download again. The paid job already exists.
            try:
                state = batch.get_job_details(job_id)["state"]
                while state in {"queued", "processing"}:
                    time.sleep(5)
                    state = batch.get_job_details(job_id)["state"]
                if state != "done":
                    raise RuntimeError(f"Batch {job_id} is {state}")
                with TemporaryDirectory(dir=raw_dir) as folder:
                    files = batch.download(job_id=job_id, output_dir=folder)
                    dbns = [p for p in files if p.name.endswith(".dbn.zst")]
                    parts = []
                    for dbn_file in dbns:  # An existing job may have split its output.
                        with dbn_file.open("rb") as stream:
                            parts.append(db.DBNStore.from_bytes(stream).to_df(price_type="fixed", pretty_ts=False))
                    raw = pd.concat(parts).reset_index()
                    info = dict(request=request, job_id=job_id, rows=len(raw), price_scale_usd=1e-9, undef_price=2**63-1,
                                version=1, compressed_dbn_bytes=sum(p.stat().st_size for p in dbns),
                                timestamps="UTC Unix ns; ts_recv = interval end, ts_event = last trade")
                    table = pa.Table.from_pandas(raw, preserve_index=False)
                    table = table.replace_schema_metadata({**table.schema.metadata, RAW_META: json.dumps(info).encode()})
                    tmp = Path(folder) / "raw.parquet"
                    pq.write_table(table, tmp, compression="zstd")
                    tmp.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return paths


def build_hour_bar_spread(month="2025-01", *, output_dir=OUTPUT_DIR):
    """Sample the last same-session BBO at each decision and write monthly/YYYY-MM.parquet."""
    period = pd.Period(month, freq="M")
    calendar = xcals.get_calendar("XNYS", start=period.start_time, end=period.end_time.normalize())
    frames = []
    for day, session in calendar.schedule.iterrows():
        path = Path(output_dir) / "raw" / f"{day.date()}.parquet"
        info = json.loads(pq.read_schema(path).metadata[RAW_META])
        tickers = get_universe(day.date())
        request = info["request"]
        if (request["symbols"] != tickers or request["dataset"] != "XNAS.ITCH" or request["schema"] != "bbo-1m"
                or pd.Timestamp(request["start"]) != session.open + MINUTE or pd.Timestamp(request["end"]) != session.close + MINUTE):
            raise ValueError(f"Raw BBO file does not match this session: {path}")
        book = pd.read_parquet(path, columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00", *DEPTH])
        book = book.rename(columns={"symbol": "ticker", "bid_px_00": "bid", "ask_px_00": "ask"})
        book["source_ts"] = pd.to_datetime(book.pop("ts_recv"), utc=True, unit="ns").dt.tz_convert(NY)
        book = book.loc[book["source_ts"].between(session.open, session.close, inclusive="right")].copy()
        prices = book[["bid", "ask"]]
        book[["bid", "ask"]] = prices.mask(prices == 2**63-1).astype(float) / 1e9
        book[DEPTH] = book[DEPTH].astype("UInt32")
        hours = pd.date_range(session.open.tz_convert(NY).ceil("h"), session.close.tz_convert(NY), freq="h", inclusive="left")
        hours = hours[(hours.hour >= 10) & (hours.hour <= 15)].as_unit("ns")
        grid = pd.MultiIndex.from_product([hours, tickers], names=["ts", "ticker"]).to_frame(index=False)
        book["ticker"] = book["ticker"].astype(grid["ticker"].dtype)
        rows = pd.merge_asof(grid, book.sort_values("source_ts"), left_on="ts", right_on="source_ts",
                             by="ticker", direction="backward")
        rows["age_seconds"] = (rows["ts"] - rows["source_ts"]).dt.total_seconds()
        rows["carried"] = rows["age_seconds"] > 0
        bid, ask = rows["bid"], rows["ask"]
        both = (bid > 0) & (ask > 0)
        rows["mid"] = ((bid + ask) / 2).where(both)
        rows["spread"] = (ask - bid).round(9).where(both & (ask >= bid))
        rows["half_spread"] = rows["spread"] / 2
        rows["spread_bps"] = 10_000 * rows["spread"] / rows["mid"]
        rows["half_spread_bps"] = rows["spread_bps"] / 2
        wide = (100 * rows["spread"] - rows["mid"]).round(10) > 0  # strictly above 100 bps
        rows["status"] = np.select([rows["source_ts"].isna(), ~(bid.gt(0) | ask.gt(0)), ~both, ask < bid, ask == bid, wide],
                                   ["missing", "no_valid_side", "one_sided", "crossed", "locked", "wide"], default="ok")
        frames.append(rows)
    path = Path(output_dir) / "monthly" / f"{period}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as folder:
        tmp = Path(folder) / "hourly.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(path)
    return path


if __name__ == "__main__":
    download_spread_data()
    print(build_hour_bar_spread())
