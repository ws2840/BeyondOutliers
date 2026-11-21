"""Offline regression checks for Nasdaq decision-time spreads."""

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import databento as db
import databento_dbn as dbn
import pandas as pd

from Python.market_data import spread


def ny(time):
    return pd.Timestamp("2018-05-01 " + time, tz="America/New_York")


def sdk_quotes():
    # The second record arrives after the decision despite an old last-trade stamp.
    trade_time = ny("09:45").value
    records = [
        dbn.BBOMsg(
            rtype=dbn.RType.BBO_1M, publisher_id=2, instrument_id=7,
            ts_event=trade_time, ts_recv=ny(minute).value, price=50_000_000_000,
            size=10, side=dbn.Side.NONE, levels=dbn.BidAskPair(bid, ask),
        )
        for minute, bid, ask in [
            ("10:00", 49_000_000_000, 51_000_000_000),
            ("10:01", 50_000_000_000, 50_500_000_000),
        ]
    ]
    mapping = SimpleNamespace(
        start_date=date(2018, 5, 1), end_date=date(2018, 5, 2), symbol="7",
    )
    metadata = dbn.Metadata(
        dataset="XNAS.ITCH", start=trade_time, schema=dbn.Schema.BBO_1M,
        stype_in=dbn.SType.RAW_SYMBOL, stype_out=dbn.SType.INSTRUMENT_ID,
        symbols=["AAPL"], mappings=[SimpleNamespace(raw_symbol="AAPL", intervals=[mapping])],
    )
    return db.DBNStore.from_bytes(metadata.encode() + b"".join(map(bytes, records)))


class SpreadTests(unittest.TestCase):
    def test_minute_bbo_matches_last_tick_before_each_interval_end(self):
        # Synthetic AAPL book updates in exchange receive order. Each tuple is a whole BBO.
        ticks = [
            (ny("09:59:00"), 48, 51),
            (ny("09:59:59.999999999"), 49, 51),
            (ny("10:00:00"), 50, 51),
            (ny("10:00:59.999999999"), 50, 50.5),
            (ny("10:01:00"), 51, 52),
        ]
        ends = pd.DatetimeIndex([ny("10:00"), ny("10:01")])
        observed = spread.spread_snapshot(sdk_quotes().to_df(), ["AAPL"], ends, ny("09:30"))
        for row in observed.itertuples(index=False):
            start = row.ts - pd.offsets.Minute()
            last = next(tick for tick in reversed(ticks) if start <= tick[0] < row.ts)
            with self.subTest(interval_end=row.ts):
                self.assertEqual((row.bid, row.ask), last[1:])
                self.assertEqual(row.spread, last[2] - last[1])

    def test_sdk_bbo_uses_interval_end_and_ignores_future_update(self):
        times = pd.DatetimeIndex([ny("10:00"), ny("11:00")])
        quotes = sdk_quotes().to_df()
        self.assertEqual(quotes.index.name, "ts_recv")
        result = spread.spread_snapshot(quotes, ["AAPL"], times, ny("09:30"))
        first, second = result.itertuples(index=False)
        self.assertEqual((first.bid, first.ask, first.source_ts), (49, 51, ny("10:00")))
        self.assertEqual((first.mid, first.spread, first.half_spread), (50, 2, 1))
        self.assertEqual((first.spread_bps, first.half_spread_bps), (400, 200))
        self.assertEqual((first.age_seconds, first.carried), (0, False))
        self.assertEqual((second.bid, second.ask, second.source_ts), (50, 50.5, ny("10:01")))
        self.assertEqual((second.age_seconds, second.carried), (59 * 60, True))

    def test_snapshot_never_reuses_another_symbol_or_preopen_book(self):
        quotes = pd.DataFrame([
            (ny("09:59"), "A", 99, 100),
            (ny("16:00") - pd.Timedelta(days=1), "B", 19, 20),
            (ny("09:30"), "B", 29, 30),
            (ny("10:01"), "B", 39, 40),
        ], columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"])
        times = pd.DatetimeIndex([ny("10:00")])
        result = spread.spread_snapshot(quotes, ["A", "B"], times, ny("09:30")).set_index("ticker")
        self.assertEqual(result.loc["A", "age_seconds"], 60)
        self.assertEqual(result.loc["B", "status"], "missing")
        self.assertTrue(pd.isna(result.loc["B", "source_ts"]))
        self.assertTrue(pd.isna(result.loc["B", "spread"]))

    def test_quality_flags_and_costs(self):
        sides = {"ONE": (9, None), "EMPTY": (0, 0), "LOCK": (10, 10),
                 "CROSS": (11, 10), "WIDE": (9, 11)}
        quotes = pd.DataFrame([(ny("10:00"), symbol, *s) for symbol, s in sides.items()],
                              columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"])
        times = pd.DatetimeIndex([ny("10:00")])
        result = spread.spread_snapshot(quotes, list(sides), times, ny("09:30")).set_index("ticker")
        self.assertEqual(result["status"].to_dict(), {
            "ONE": "one_sided", "EMPTY": "no_valid_side", "LOCK": "locked",
            "CROSS": "crossed", "WIDE": "wide",
        })
        self.assertTrue(result.loc[["ONE", "EMPTY", "CROSS"],
                                   ["spread", "half_spread", "spread_bps", "half_spread_bps"]].isna().all().all())
        self.assertEqual(result.loc["LOCK", "spread_bps"], 0)
        self.assertEqual(result.loc["WIDE", "spread_bps"], 2000)

    def test_holiday_half_day_and_rerun(self):
        client = Mock()
        client.timeseries.get_range.return_value.to_df.return_value = pd.DataFrame()
        with TemporaryDirectory() as folder, patch.object(spread, "get_universe", return_value=["AAPL"]):
            def build(start, end):
                return spread.build_spreads(start, end, output_dir=Path(folder), client=client)

            self.assertEqual(build("2018-05-28", "2018-05-29"), [])
            client.timeseries.get_range.assert_not_called()
            paths = build("2018-07-03", "2018-07-04")
            self.assertEqual(len(paths), 1)
            saved = pd.read_parquet(paths[0])
            self.assertEqual(saved["ts"].dt.strftime("%H:%M").tolist(), ["10:00", "11:00", "12:00"])
            self.assertEqual(saved["status"].tolist(), ["missing"] * 3)
            self.assertEqual(build("2018-07-03", "2018-07-04"), paths)
            client.timeseries.get_range.assert_called_once()

    def test_size_and_metadata_use_the_download_window(self):
        symbols = ["AAPL", "MSFT"]
        dates = ("2018-05-01", "2018-05-02")
        metadata = Mock()
        metadata.get_record_count.return_value = 600
        metadata.get_billable_size.return_value = 48_000
        metadata.get_cost.return_value = 0.001
        download = Mock()
        download.timeseries.get_range.return_value = sdk_quotes()
        with TemporaryDirectory() as folder, patch.object(spread, "get_universe", return_value=symbols), \
                patch.object(spread, "_historical_client", side_effect=AssertionError("offline")):
            size = spread.spread_size(*dates).iloc[0]
            estimate = spread.estimate_spreads(*dates, client=SimpleNamespace(metadata=metadata)).iloc[0]
            spread.build_spreads(*dates, output_dir=Path(folder), client=download)
        query = download.timeseries.get_range.call_args.kwargs
        self.assertEqual((query["dataset"], query["schema"], query["symbols"]), ("XNAS.ITCH", "bbo-1m", symbols))
        self.assertEqual((query["start"], query["end"]), (ny("09:31"), ny("15:01")))
        self.assertEqual((size.minutes, size.stocks, size.stock_decisions, size.max_record_bytes),
                         (330, 2, 12, 330 * 2 * 80))
        for method in (metadata.get_record_count, metadata.get_billable_size, metadata.get_cost):
            method.assert_called_once_with(**query)
        self.assertEqual((estimate.records, estimate.billable_bytes, estimate.cost_usd), (600, 48_000, 0.001))
