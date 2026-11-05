"""Replay target positions for one simulated stock using bid/ask quotes."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path


SAMPLE_PATH = Path(__file__).with_name("sample_quotes.csv")
INITIAL_CASH = Decimal("2000.00")


@dataclass(frozen=True)
class Quote:
    time_et: str
    bid: Decimal
    ask: Decimal
    target_qty: int
    max_fill_qty: int

    def __post_init__(self) -> None:
        if not (self.bid.is_finite() and self.ask.is_finite() and 0 < self.bid <= self.ask):
            raise ValueError(f"Expected finite prices with 0 < bid <= ask at {self.time_et}")
        if self.max_fill_qty < 0:
            raise ValueError(f"Negative fill capacity at {self.time_et}")


@dataclass(frozen=True)
class Snapshot:
    time_et: str
    mid: Decimal
    target_qty: int
    order_qty: int
    fill_qty: int
    fill_price: Decimal | None
    position_qty: int
    cash: Decimal
    equity: Decimal
    gross_pnl: Decimal
    spread_cost: Decimal
    net_pnl: Decimal


def load_quotes(path: Path) -> list[Quote]:
    with path.open(encoding="utf-8", newline="") as source:
        return [
            Quote(
                time_et=row["time_et"],
                bid=Decimal(row["bid"]),
                ask=Decimal(row["ask"]),
                target_qty=int(row["target_qty"]),
                max_fill_qty=int(row["max_fill_qty"]),
            )
            for row in csv.DictReader(source)
        ]


def replay(quotes: Iterable[Quote], initial_cash: Decimal = INITIAL_CASH) -> list[Snapshot]:
    """Replay strictly increasing ISO timestamps with UTC offsets. Each order expires at its timestamp."""
    cash = initial_cash
    position_qty = 0
    gross_pnl = Decimal(0)
    spread_cost = Decimal(0)
    previous_mid = None
    previous_time = None
    snapshots = []

    for quote in quotes:
        timestamp = datetime.fromisoformat(quote.time_et)
        if timestamp.tzinfo is None:
            raise ValueError(f"Quote timestamp must include a UTC offset: {quote.time_et}")
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError(f"Quote timestamps must be strictly increasing: {quote.time_et}")

        mid = (quote.bid + quote.ask) / 2
        # Only shares actually held over the preceding interval earn its price change.
        if previous_mid is not None:
            gross_pnl += position_qty * (mid - previous_mid)

        order_qty = quote.target_qty - position_qty
        fill_qty = max(-quote.max_fill_qty, min(order_qty, quote.max_fill_qty))
        fill_price = None
        if fill_qty:
            fill_price = quote.ask if fill_qty > 0 else quote.bid
            cash -= fill_qty * fill_price
            position_qty += fill_qty
            # Cash already reflects the execution price. Track its cost for attribution.
            spread_cost += fill_qty * (fill_price - mid)

        equity = cash + position_qty * mid
        snapshots.append(
            Snapshot(
                time_et=quote.time_et,
                mid=mid,
                target_qty=quote.target_qty,
                order_qty=order_qty,
                fill_qty=fill_qty,
                fill_price=fill_price,
                position_qty=position_qty,
                cash=cash,
                equity=equity,
                gross_pnl=gross_pnl,
                spread_cost=spread_cost,
                net_pnl=equity - initial_cash,
            )
        )
        previous_mid = mid
        previous_time = timestamp
    return snapshots


def _amount(value: Decimal) -> str:
    places = max(2, -value.normalize().as_tuple().exponent)
    return f"{value:.{places}f}"


def _signed(qty: int) -> str:
    return f"{qty:+d}" if qty else "0"


def main() -> None:
    print(f"Simulated stock | times in ET | initial cash ${_amount(INITIAL_CASH)}")
    print("All monetary values are in USD. P&L and spread cost are cumulative.")
    for step in replay(load_quotes(SAMPLE_PATH)):
        fill = "0 (no fill)"
        if step.fill_price is not None:
            fill = f"{_signed(step.fill_qty)} @ {_amount(step.fill_price)}"
        print(f"\n{step.time_et}  mid={_amount(step.mid)}")
        print(
            f"  target={_signed(step.target_qty)}  order={_signed(step.order_qty)}  "
            f"fill={fill}  expired={_signed(step.order_qty - step.fill_qty)}  "
            f"position={_signed(step.position_qty)}"
        )
        print(
            f"  cash={_amount(step.cash)}  equity={_amount(step.equity)}  "
            f"gross_pnl={_amount(step.gross_pnl)}  spread_cost={_amount(step.spread_cost)}  "
            f"net_pnl={_amount(step.net_pnl)}"
        )


if __name__ == "__main__":
    main()
