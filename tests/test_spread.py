"""Offline tests for full-session downloads and decision-hour snapshots."""
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import databento as db
import databento_dbn as dbn
import pandas as pd

from Python.market_data import spread


def synthetic_dbn(day, quotes):
    """Make real BBO-1m DBN bytes, with a deliberately old last-trade timestamp."""
    def stamp(time):
        return pd.Timestamp(f"{day} {time}", tz="America/New_York").value

    symbols = ["AAPL", "MSFT"]
    mappings = []
    records = []
    for number, symbol in enumerate(symbols, start=1):
        interval = SimpleNamespace(start_date=day, end_date=day + timedelta(days=1), symbol=str(number))
        mappings.append(SimpleNamespace(raw_symbol=symbol, intervals=[interval]))
    for time, symbol, bid, ask in quotes:
        records.append(dbn.BBOMsg(
            rtype=dbn.RType.BBO_1M, publisher_id=2, instrument_id=symbols.index(symbol) + 1,
            ts_recv=stamp(time), ts_event=stamp("09:31"), price=50_000_000_000,
            size=10, side=dbn.Side.NONE, levels=dbn.BidAskPair(bid, ask, 123, 456, 2, 3),
        ))
    metadata = dbn.Metadata(
        dataset="XNAS.ITCH", schema=dbn.Schema.BBO_1M, start=stamp("09:31"),
        stype_in=dbn.SType.RAW_SYMBOL, stype_out=dbn.SType.INSTRUMENT_ID, mappings=mappings,
    )
    return db.DBNStore.from_bytes(metadata.encode() + b"".join(map(bytes, records)))


class SpreadTests(unittest.TestCase):
    def test_closed_sessions_never_order_data(self):
        for day, next_day in [("2018-05-28", "2018-05-29"), ("2025-01-09", "2025-01-10")]:
            with self.subTest(day=day), TemporaryDirectory() as folder:
                result = spread.download_spread_data(day, next_day, output_dir=folder, client=object())
                self.assertEqual(result, [])

    def test_submit_retries_only_rate_limits(self):
        batch = Mock()
        batch.list_jobs.return_value = []
        batch.submit_job.side_effect = [db.BentoClientError(429, headers={"Retry-After": "12"}), TimeoutError()]
        client = SimpleNamespace(batch=batch)
        with TemporaryDirectory() as folder, patch.object(spread.time, "sleep") as sleep:
            with self.assertRaises(TimeoutError):
                spread.download_spread_data("2025-01-02", "2025-01-03", output_dir=folder, client=client)
            sleep.assert_called_once_with(12)
            with self.assertRaisesRegex(RuntimeError, "Unconfirmed batch"):
                spread.download_spread_data("2025-01-02", "2025-01-03", output_dir=folder, client=client)
            self.assertEqual(batch.submit_job.call_count, 2)
            self.assertEqual(*batch.submit_job.call_args_list)

    def test_download_cache_and_hourly_book(self):
        # Two adjacent sessions include a normal close and the July 3 early close.
        schedule = spread.xcals.get_calendar("XNYS").schedule.loc["2018-07-02":"2018-07-03"]
        fixture_quotes = [
            [("10:00", "AAPL", 49_000_000_000, 51_000_000_000),
             ("10:00", "MSFT", 20_000_000_000, 20_000_000_000),
             ("10:01", "AAPL", 50_000_000_000, 50_500_000_000)],
            [("10:00", "AAPL", 51_000_000_000, dbn.UNDEF_PRICE),
             ("11:00", "AAPL", 52_000_000_000, 51_000_000_000)],
        ]
        with (
            TemporaryDirectory() as folder,
            patch.object(spread, "get_universe", return_value=["AAPL", "MSFT"]),
            patch.object(spread.xcals, "get_calendar", return_value=SimpleNamespace(schedule=schedule)),
        ):
            root = Path(folder)
            stores = {}
            files = {}
            for stamp, rows in zip(schedule.index, fixture_quotes):
                job = str(stamp.date())
                stores[job] = synthetic_dbn(stamp.date(), rows)
                files[job] = root / f"{job}.dbn.zst"
                stores[job].to_file(files[job], compression="zstd")

            batch = Mock()
            batch.list_jobs.return_value = []
            batch.submit_job.side_effect = lambda **query: {"id": str(pd.Timestamp(query["start"]).date())}
            batch.get_job_details.return_value = {"state": "done"}
            batch.download.side_effect = lambda job_id, **kwargs: [files[job_id]]

            client = SimpleNamespace(batch=batch)
            days = spread.download_spread_data("2018-07-01", "2018-08-01", output_dir=root, client=client)
            self.assertEqual(batch.submit_job.call_count, 2)
            for path in days:
                raw = pd.read_parquet(path)
                expected = stores[path.stem].to_df(price_type="fixed", pretty_ts=False).reset_index()
                pd.testing.assert_frame_equal(raw[expected.columns], expected, check_dtype=False)
            cached = spread.download_spread_data("2018-07-01", "2018-08-01", output_dir=root, client=object())
            self.assertEqual(days, cached)

            days[0].unlink()
            spread.download_spread_data("2018-07-01", "2018-08-01", output_dir=root, client=client)
            self.assertTrue(days[0].exists())
            self.assertEqual(batch.submit_job.call_count, 2)

            orders = {}
            for call, raw_path in zip(batch.submit_job.call_args_list, days):
                orders[raw_path.stem] = dict(call.kwargs, id=raw_path.stem, state="done", stype_out="instrument_id")
            batch.list_jobs.return_value = [{"id": job_id} for job_id in orders]
            batch.get_job_details.side_effect = orders.__getitem__
            other_machine = spread.download_spread_data("2018-07-01", "2018-08-01", output_dir=root / "new", client=client)
            self.assertEqual([f.name for f in other_machine], [f.name for f in days])
            self.assertEqual(batch.submit_job.call_count, 2)
            month = pd.read_parquet(spread.build_hour_bar_spread("2018-07", output_dir=root))

        self.assertEqual(month.groupby(month.ts.dt.day).size().to_dict(), {2: 12, 3: 6})
        aapl = month.loc[month.ticker.eq("AAPL")].reset_index(drop=True)
        first = aapl.iloc[0]
        self.assertEqual((first.bid, first.ask, first.spread, first.status), (49, 51, 2, "wide"))
        self.assertEqual((first.bid_sz_00, first.ask_sz_00, first.bid_ct_00, first.ask_ct_00), (123, 456, 2, 3))
        later = aapl.iloc[1]
        self.assertEqual((later.bid, later.ask, later.age_seconds), (50, 50.5, 59 * 60))
        self.assertEqual(aapl.iloc[6:].status.tolist(), ["one_sided", "crossed", "crossed"])
        msft = month.loc[month.ticker.eq("MSFT")]
        self.assertTrue(msft.loc[msft.ts.dt.day.eq(2), "status"].eq("locked").all())
        self.assertTrue(msft.loc[msft.ts.dt.day.eq(3), "source_ts"].isna().all())
