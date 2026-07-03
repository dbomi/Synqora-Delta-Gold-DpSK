"""
=============================================================================
POSITION MANAGER — SYNQORA DELTA GOLD FABLE
Minimal, simulation-faithful management:
  - Hard SL/TP are set at entry (broker side) — no trailing interference.
  - Negative time stop: close positions still losing after N primary bars.
  - Max-hold time stop: force close after MAX_HOLD_BARS primary bars.
The validated walk-forward results came from fixed TP/SL geometry; this
manager deliberately does NOT trail, partial-close, or ladder.
=============================================================================
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from config import (
    SYMBOL, MAGIC_NUMBER, PRIMARY_TF,
    MAX_HOLD_BARS, USE_NEGATIVE_TIME_STOP, NEGATIVE_TIME_STOP_BARS,
    USE_EQUITY_TIERED_BREAKEVEN, BREAKEVEN_EQUITY_CUTOFF,
    BREAKEVEN_ARM_R, BREAKEVEN_BUFFER_POINTS,
)
from execution_engine import close_position, modify_position_sl

logger = logging.getLogger("PositionManager")

TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}
_BAR_MINUTES = TF_MINUTES.get(PRIMARY_TF, 15)


def _bars_held(open_time_epoch: float, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    opened = datetime.fromtimestamp(open_time_epoch, tz=timezone.utc)
    return (now - opened).total_seconds() / 60.0 / _BAR_MINUTES


def _apply_equity_tiered_breakeven(pos, symbol: str) -> None:
    """
    Survival-phase protection: while account equity is below
    BREAKEVEN_EQUITY_CUTOFF, once a position's profit reaches
    BREAKEVEN_ARM_R × R (R = entry-to-original-SL distance, recovered from
    the TP which sits at 2R), move the SL to entry ± a small buffer.

    One-way: the SL is only ever tightened, so positions armed while the
    account was small keep their protection even if equity later grows
    past the cutoff mid-trade. Above the cutoff, new arming stops and the
    validated fixed TP/SL geometry runs untouched.
    """
    if not USE_EQUITY_TIERED_BREAKEVEN:
        return

    acc = mt5.account_info()
    equity = float(getattr(acc, "equity", 0.0) or 0.0) if acc else 0.0
    if equity <= 0 or equity >= BREAKEVEN_EQUITY_CUTOFF:
        return

    info = mt5.symbol_info(symbol)
    point = float(getattr(info, "point", 0.01) or 0.01) if info else 0.01
    entry = float(pos.price_open)
    tp    = float(pos.tp)
    if tp <= 0:
        return
    # Original R from the fixed geometry: TP = entry ± 2R.
    r = abs(tp - entry) / 2.0
    if r <= 0:
        return

    is_buy  = pos.type == mt5.POSITION_TYPE_BUY
    price   = float(pos.price_current)
    fav     = (price - entry) if is_buy else (entry - price)
    if fav < BREAKEVEN_ARM_R * r:
        return

    buffer = BREAKEVEN_BUFFER_POINTS * point
    be_sl  = entry + buffer if is_buy else entry - buffer
    cur_sl = float(pos.sl)
    already = cur_sl > 0 and ((cur_sl >= be_sl) if is_buy else (cur_sl <= be_sl))
    if already:
        return

    logger.info(f"[PM] Breakeven armed (equity {equity:.2f} < "
                f"{BREAKEVEN_EQUITY_CUTOFF:.0f}): ticket={pos.ticket} "
                f"profit={fav / r:.2f}R -> SL {cur_sl:.2f} -> {be_sl:.2f}")
    modify_position_sl(pos.ticket, be_sl, symbol)


def manage_open_positions(symbol: str = SYMBOL) -> int:
    """
    Apply time-based exits and equity-tiered breakeven protection to all
    Fable positions on this symbol. Returns the number of positions closed.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return 0

    closed = 0
    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue

        _apply_equity_tiered_breakeven(pos, symbol)

        held = _bars_held(pos.time)

        if held >= MAX_HOLD_BARS:
            logger.info(f"[PM] Max-hold stop: ticket={pos.ticket} "
                        f"held={held:.1f} bars >= {MAX_HOLD_BARS} (pnl={pos.profit:.2f})")
            if close_position(pos.ticket, symbol, comment="FableMaxHold"):
                closed += 1
            continue

        if USE_NEGATIVE_TIME_STOP and held >= NEGATIVE_TIME_STOP_BARS and pos.profit < 0:
            logger.info(f"[PM] Negative time stop: ticket={pos.ticket} "
                        f"held={held:.1f} bars, pnl={pos.profit:.2f} < 0")
            if close_position(pos.ticket, symbol, comment="FableNegTimeStop"):
                closed += 1

    return closed


def open_position_counts(symbol: str = SYMBOL) -> dict:
    """Count open Fable positions by side."""
    positions = mt5.positions_get(symbol=symbol)
    counts = {"BUY": 0, "SELL": 0}
    if positions:
        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue
            if pos.type == mt5.POSITION_TYPE_BUY:
                counts["BUY"] += 1
            else:
                counts["SELL"] += 1
    return counts


def open_positions_frame(symbol: str = SYMBOL) -> pd.DataFrame:
    """Fable positions as a DataFrame (for the meta-agent gate)."""
    positions = mt5.positions_get(symbol=symbol)
    rows = []
    if positions:
        for p in positions:
            if p.magic != MAGIC_NUMBER:
                continue
            rows.append({
                "ticket":  p.ticket,
                "type":    "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume":  p.volume,
                "profit":  p.profit,
                "open_time": pd.to_datetime(p.time, unit="s", utc=True),
            })
    return pd.DataFrame(rows)
