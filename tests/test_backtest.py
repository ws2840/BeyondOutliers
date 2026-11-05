import unittest
from decimal import Decimal

from Python.main import SAMPLE_PATH, Quote, load_quotes, replay


class BacktestTests(unittest.TestCase):
    def test_sample_ledger(self):
        ledger = replay(load_quotes(SAMPLE_PATH))

        self.assertEqual(
            [(row.target_qty, row.order_qty, row.fill_qty, row.position_qty) for row in ledger],
            [(10, 10, 10, 10), (6, -4, 0, 10), (0, -10, -10, 0)],
        )
        self.assertEqual(
            [(row.cash, row.equity) for row in ledger],
            [
                (Decimal("999.80"), Decimal("1999.80")),
                (Decimal("999.80"), Decimal("2009.80")),
                (Decimal("2000.00"), Decimal("2000.00")),
            ],
        )
        self.assertEqual(
            [(row.gross_pnl, row.spread_cost, row.net_pnl) for row in ledger],
            [
                (Decimal("0.00"), Decimal("0.20"), Decimal("-0.20")),
                (Decimal("10.00"), Decimal("0.20"), Decimal("9.80")),
                (Decimal("0.50"), Decimal("0.50"), Decimal("0.00")),
            ],
        )
        self.assertIsNone(ledger[1].fill_price)

    def test_rejects_out_of_order_quotes(self):
        # During the fall clock change, 01:45 EDT comes before 01:15 EST.
        quotes = [
            Quote("2026-11-01T01:15:00-05:00", Decimal("10.00"), Decimal("10.02"), 1, 1),
            Quote("2026-11-01T01:45:00-04:00", Decimal("10.00"), Decimal("10.02"), 0, 1),
        ]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            replay(quotes)

    def test_partial_fill_then_changed_target(self):
        quotes = [
            Quote("2026-11-03T09:45:00-05:00", Decimal("9.99"), Decimal("10.01"), 5, 2),
            Quote("2026-11-03T10:15:00-05:00", Decimal("11.99"), Decimal("12.01"), 3, 10),
            Quote("2026-11-03T10:45:00-05:00", Decimal("12.00"), Decimal("14.00"), 3, 10),
        ]
        ledger = replay(quotes)

        self.assertEqual(
            [(row.order_qty, row.fill_qty, row.position_qty) for row in ledger],
            [(5, 2, 2), (1, 1, 3), (0, 0, 3)],
        )
        self.assertEqual(
            [row.fill_price for row in ledger],
            [Decimal("10.01"), Decimal("12.01"), None],
        )
        self.assertEqual(
            [row.spread_cost for row in ledger],
            [Decimal("0.02"), Decimal("0.03"), Decimal("0.03")],
        )
        self.assertEqual(ledger[-1].cash, Decimal("1967.97"))
        self.assertEqual(ledger[-1].net_pnl, Decimal("6.97"))


if __name__ == "__main__":
    unittest.main()
