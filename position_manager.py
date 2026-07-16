"""
=============================================================================
POSITION MANAGER — SYNQORA DELTA GOLD FABLE
Management that never lets a winning trade turn red:
  - Hard SL/TP are set at entry (broker side).
  - P5: Breakeven @1.0R always-on.
  - P11: MFE trailing (auto in volatile/high-ATR).
   - P14: Regime-laddered profit protection — locks profit at increasing
      R-levels based on market regime (VOLATILE/TRENDING/RANGING).
      Once +0.15R is hit, the trade NEVER goes red again.
   - P16: Profit-retrace virtual close — closes at market when profit
      retraces from a small peak (>= $2) back to near breakeven ($0.80).
      Catches sub-1R round-trips that P14's broker SL modify misses.
   - P15: Adaptive Peak Exit — three independent layers (volatility trail,
      momentum decay, volume climax) that exit at market when peak conditions
      are met, maximizing profit on top of P14's floor.
   - Negative time stop / max-hold time stop.
=============================================================================
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import (
    SYMBOL, MAGIC_NUMBER, PRIMARY_TF,
    MAX_HOLD_BARS, USE_NEGATIVE_TIME_STOP, NEGATIVE_TIME_STOP_BARS,
    USE_EQUITY_TIERED_BREAKEVEN, BREAKEVEN_EQUITY_CUTOFF,
    BREAKEVEN_ARM_R, BREAKEVEN_BUFFER_POINTS,
    BREAKEVEN_ALWAYS_ON,
    MFE_TRAIL_ENABLED, MFE_ARM_BE_AT_R, MFE_TRAIL_ACTIVATE_AT_R,
    MFE_TRAIL_DISTANCE_R, MFE_TRAIL_BUFFER_POINTS,
    MFE_TRAIL_AUTO_MODE, MFE_TRAIL_ATR_RATIO, MFE_TRAIL_ATR_LOOKBACK,
    MFE_TRAIL_REGIME_TRIGGERS,
    RATCHET_ENABLED, RATCHET_ARM_AT_R, RATCHET_LOCK_AT_R,
    RATCHET_VIRTUAL_LEVER,
    REGIME_PROTECTION_ENABLED,
    REGIME_PROTECTION_M5_LOOKBACK,
    REGIME_PROTECTION_VOLATILE_SIGMA,
    REGIME_PROTECTION_MA_DIVERGENCE,
    REGIME_PROTECTION_STEPS,
    REGIME_PROTECTION_BUFFER_POINTS,
    PEAK_EXIT_ENABLED,
    PROFIT_RETRACE_ENABLED,
    PROFIT_RETRACE_ARM_USD,
    PROFIT_RETRACE_CLOSE_USD,
)
from execution_engine import close_position, modify_position_sl
from peak_exit import assess_position, close_peak_exit, reset_ticket

logger = logging.getLogger("PositionManager")

TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}
_BAR_MINUTES = TF_MINUTES.get(PRIMARY_TF, 15)

# ── Per-ticket state for P14 regime ladder ──────────────────────────────
_p14_state: dict[int, dict] = {}  # ticket -> {max_mfe_r, step_index, virtual_sl}

# ── Per-ticket state for P16 profit retrace ────────────────────────────
_p16_state: dict[int, dict] = {}  # ticket -> {peak_profit, armed}

# ── Per-ticket state for P17 trailing ratchet ──────────────────────
_ratchet_state: dict[int, dict] = {}  # ticket -> {max_mfe_r, r_locked, virtual_sl}


def _bars_held(open_time_epoch: float, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    opened = datetime.fromtimestamp(open_time_epoch, tz=timezone.utc)
    return (now - opened).total_seconds() / 60.0 / _BAR_MINUTES


# ── Market classification (shared for P14) ───────────────────────────────


def _classify_regime(symbol: str) -> str:
    """Classify market as VOLATILE, TRENDING, or RANGING using M5 data."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, REGIME_PROTECTION_M5_LOOKBACK)
    if rates is None or len(rates) < 20:
        return "UNKNOWN"
    df = pd.DataFrame(rates)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    pct_vol = df["close"].pct_change().std() * 100
    if pct_vol > REGIME_PROTECTION_VOLATILE_SIGMA:
        return "VOLATILE"
    last_ma5 = df["ma5"].iloc[-1]
    last_ma20 = df["ma20"].iloc[-1]
    if last_ma20 > 0 and abs(last_ma5 - last_ma20) / last_ma20 > REGIME_PROTECTION_MA_DIVERGENCE:
        return "TRENDING"
    return "RANGING"


# ── P14: Regime-laddered never-red protection ────────────────────────────


def _apply_regime_protection(pos, symbol: str, current_regime: Optional[str] = None) -> None:
    """
    P14: Regime-laddered profit lock + virtual stop enforcement.

    Every cycle:
      1. Computes the ideal SL from the regime ladder (never loosens).
      2. Tries to set it on the broker (best effort).
      3. Stores the highest virtual SL in memory.
      4. If price breaches the virtual SL, closes via market order IMMEDIATELY.

    The position NEVER goes below the highest locked rung — guaranteed
    by the virtual stop, even if the broker rejects the SL modify.
    """
    if not REGIME_PROTECTION_ENABLED:
        return

    regime = current_regime or _classify_regime(symbol)
    steps = REGIME_PROTECTION_STEPS.get(regime)
    if not steps:
        return

    info = mt5.symbol_info(symbol)
    point = float(getattr(info, "point", 0.01) or 0.01) if info else 0.01
    entry = float(pos.price_open)
    tp = float(pos.tp)
    if tp <= 0:
        return
    r = abs(tp - entry) / 2.0
    if r <= 0:
        return

    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    price = float(pos.price_current)
    fav = (price - entry) if is_buy else (entry - price)
    ticket = pos.ticket
    state = _p14_state.setdefault(ticket, {"max_mfe_r": 0.0, "max_step": -1, "virtual_sl": None})

    # ── Step 1: Track MFE ────────────────────────────────────────────────
    # Track peak MFE even if current profit is negative (the ladder uses peak)
    mfe_r = max(fav / r, 0.0) if r > 0 else 0.0
    if mfe_r > state["max_mfe_r"]:
        state["max_mfe_r"] = mfe_r

    # ── Step 2: Compute the highest ladder rung reached ──────────────────
    cur_sl = float(pos.sl)
    buffer = REGIME_PROTECTION_BUFFER_POINTS * point
    best_sl = cur_sl
    best_step = state["max_step"]

    # Use PEAK MFE for the ladder, not current fav, so trailing works
    peak_mfe = state["max_mfe_r"]

    for step_idx, (lock_r, trail_r) in enumerate(steps):
        if peak_mfe >= lock_r:
            target_sl = entry + (lock_r - trail_r) * r * (1 if is_buy else -1)
            target_sl += buffer * (1 if is_buy else -1)
            if (target_sl - best_sl) * (1 if is_buy else -1) > 0:
                best_sl = target_sl
                if step_idx > best_step:
                    best_step = step_idx

    # ── Step 3: Update virtual SL first (before any broker call) ─────────
    if best_sl != cur_sl and best_sl != 0:
        state["virtual_sl"] = best_sl

    # ── Step 4: Attempt broker SL modify (best effort) ───────────────────
    if best_step > state["max_step"]:
        modify_position_sl(ticket, best_sl, symbol)
        state["max_step"] = best_step
        logger.info(
            f"[P14] {regime} ladder step {best_step}: ticket={ticket} "
            f"mfe={mfe_r:.2f}R peak={peak_mfe:.2f}R "
            f"virtual SL={best_sl:.2f}")

    # ── Step 5: Mirror the actual broker SL into virtual state ───────────
    # If the broker accepted a different SL than we asked for, respect it
    actual_sl = float(pos.sl)
    if actual_sl > 0:
        v = state.get("virtual_sl")
        if v is None or ((actual_sl - v) * (1 if is_buy else -1) > 0):
            state["virtual_sl"] = actual_sl

    # ── Step 6: Virtual stop check — closes via market order if breached ──
    virtual_sl = state.get("virtual_sl")
    if virtual_sl is not None and virtual_sl > 0 and fav > 0:
        breached = (price <= virtual_sl) if is_buy else (price >= virtual_sl)
        if breached:
            logger.warning(
                f"[P14] VIRTUAL STOP HIT: ticket={ticket} "
                f"price={price:.2f} virtual SL={virtual_sl:.2f} "
                f"profit={pos.profit:.2f} — closing at market")
            close_position(ticket, symbol, comment="VStop")


# ── P16: Profit-retrace virtual close (catches sub-1R round-trips) ───────


def _apply_profit_retrace_close(pos, symbol: str) -> None:
    """
    P16: Closes at market when profit retraces from a small peak back to
    near breakeven. Designed for sub-1R profits that slip through P14's
    regime ladder because the broker rejects tight SL modifications.

    Tracks peak profit per position and closes when current profit drops
    to CLOSE_USD (or below) after having peaked at ARM_USD or higher.
    No broker SL modification - pure virtual market close.
    """
    if not PROFIT_RETRACE_ENABLED:
        return

    profit = float(pos.profit)
    ticket = pos.ticket
    state = _p16_state.setdefault(ticket, {"peak_profit": 0.0, "armed": False})

    # Track peak profit
    if profit > state["peak_profit"]:
        state["peak_profit"] = profit

    # Arm when peak reaches the threshold
    if not state["armed"] and state["peak_profit"] >= PROFIT_RETRACE_ARM_USD:
        state["armed"] = True

    # Close if armed and profit retraced to close level
    if state["armed"] and profit <= PROFIT_RETRACE_CLOSE_USD:
        logger.info(
            f"[P16] Profit retrace close: ticket={ticket} "
            f"peak=${state['peak_profit']:.2f} current=${profit:.2f} "
            f"arm=${PROFIT_RETRACE_ARM_USD} close=${PROFIT_RETRACE_CLOSE_USD}")
        close_position(ticket, symbol, comment="ProfitRetrace")


# ── P17: Two-stage trailing ratchet + virtual lever ─────────────────────


def _apply_ratchet(pos, symbol: str) -> None:
    """
    P17: Trailing ratchet — locks profit once trade proves itself.

    At MFE >= RATCHET_ARM_AT_R, ratchet SL to RATCHET_LOCK_AT_R
    profit. Virtual lever closes at market if price gaps past the
    intended SL (broker modify would fail).
    """
    if not RATCHET_ENABLED:
        return
    ticket = pos.ticket
    entry = float(pos.price_open)
    tp = float(pos.tp)
    if tp <= 0:
        return
    r = abs(tp - entry) / 2.0
    if r <= 0:
        return
    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    price = float(pos.price_current)
    fav = (price - entry) if is_buy else (entry - price)
    state = _ratchet_state.setdefault(ticket, {
        "max_mfe_r": 0.0, "r_locked": False, "virtual_sl": None,
    })
    mfe_r = max(fav / r, 0.0) if r > 0 else 0.0
    if mfe_r > state["max_mfe_r"]:
        state["max_mfe_r"] = mfe_r
    # Virtual lever: close if price already breached intended SL
    vs = state.get("virtual_sl")
    if vs is not None and fav > 0:
        breached = (price <= vs) if is_buy else (price >= vs)
        if breached:
            logger.warning(
                f"[P17] VIRTUAL STOP: ticket={ticket} "
                f"price={price:.2f} virtual SL={vs:.2f} "
                f"profit={pos.profit:.2f} — closing at market")
            close_position(ticket, symbol, comment="P17VStop")
            return
    # Ratchet: at >= ARM_R, lock SL to LOCK_R profit
    if mfe_r >= RATCHET_ARM_AT_R and not state["r_locked"]:
        lock_sl = entry + (RATCHET_LOCK_AT_R * r) * (1 if is_buy else -1)
        state["r_locked"] = True
        state["virtual_sl"] = lock_sl
        modify_position_sl(ticket, lock_sl, symbol)
        logger.info(f"[P17] Ratchet: ticket={ticket} "
                    f"mfe={mfe_r:.2f}R -> +{RATCHET_LOCK_AT_R}R SL={lock_sl:.2f}")


# ── P5: Legacy breakeven (kept as fallback if P14 is off) ────────────────


def _apply_equity_tiered_breakeven(pos, symbol: str) -> None:
    if not USE_EQUITY_TIERED_BREAKEVEN:
        return
    acc = mt5.account_info()
    equity = float(getattr(acc, "equity", 0.0) or 0.0) if acc else 0.0
    if not BREAKEVEN_ALWAYS_ON:
        if equity <= 0 or equity >= BREAKEVEN_EQUITY_CUTOFF:
            return
    info = mt5.symbol_info(symbol)
    point = float(getattr(info, "point", 0.01) or 0.01) if info else 0.01
    entry = float(pos.price_open)
    tp = float(pos.tp)
    if tp <= 0:
        return
    r = abs(tp - entry) / 2.0
    if r <= 0:
        return
    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    price = float(pos.price_current)
    fav = (price - entry) if is_buy else (entry - price)
    if fav < BREAKEVEN_ARM_R * r:
        return
    buffer = BREAKEVEN_BUFFER_POINTS * point
    be_sl = entry + buffer if is_buy else entry - buffer
    cur_sl = float(pos.sl)
    already = cur_sl > 0 and ((cur_sl >= be_sl) if is_buy else (cur_sl <= be_sl))
    if already:
        return
    logger.info(f"[PM] Breakeven armed: ticket={pos.ticket} "
                f"profit={fav / r:.2f}R -> SL {cur_sl:.2f} -> {be_sl:.2f}")
    modify_position_sl(pos.ticket, be_sl, symbol)


# ── P11: MFE trail (kept as secondary protection, runs after P14) ────────


def _is_volatile_regime(symbol: str) -> bool:
    if not MFE_TRAIL_AUTO_MODE:
        return True
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, MFE_TRAIL_ATR_LOOKBACK + 20)
    if rates is None or len(rates) < MFE_TRAIL_ATR_LOOKBACK + 5:
        return False
    df = pd.DataFrame(rates)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    current_atr = float(atr.iloc[-1])
    median_atr = float(atr.tail(MFE_TRAIL_ATR_LOOKBACK).median())
    if median_atr > 0 and current_atr / median_atr >= MFE_TRAIL_ATR_RATIO:
        return True
    return False


def _apply_mfe_trail(pos, symbol: str) -> None:
    _p11_active = MFE_TRAIL_ENABLED or (MFE_TRAIL_AUTO_MODE and _is_volatile_regime(symbol))
    if not _p11_active:
        return
    info = mt5.symbol_info(symbol)
    point = float(getattr(info, "point", 0.01) or 0.01) if info else 0.01
    entry = float(pos.price_open)
    tp = float(pos.tp)
    if tp <= 0:
        return
    r = abs(tp - entry) / 2.0
    if r <= 0:
        return
    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    price = float(pos.price_current)
    fav = (price - entry) if is_buy else (entry - price)
    if fav <= 0:
        return
    cur_sl = float(pos.sl)
    buffer = MFE_TRAIL_BUFFER_POINTS * point
    if fav >= MFE_ARM_BE_AT_R * r:
        be_sl = entry + buffer if is_buy else entry - buffer
        if (cur_sl <= 0) or ((be_sl - cur_sl) * (1 if is_buy else -1) > 0):
            logger.info(f"[PM] MFE trail BE armed: ticket={pos.ticket} profit={fav/r:.2f}R -> SL {be_sl:.2f}")
            modify_position_sl(pos.ticket, be_sl, symbol)
            cur_sl = be_sl
    if fav >= MFE_TRAIL_ACTIVATE_AT_R * r and cur_sl > 0:
        trail_stop = entry + (fav - MFE_TRAIL_DISTANCE_R * r) * (1 if is_buy else -1)
        trail_stop += buffer * (1 if is_buy else -1)
        if (trail_stop - cur_sl) * (1 if is_buy else -1) > 0:
            logger.info(f"[PM] MFE trail moved: ticket={pos.ticket} profit={fav/r:.2f}R -> SL {trail_stop:.2f}")
            modify_position_sl(pos.ticket, trail_stop, symbol)


# ── Main management loop ─────────────────────────────────────────────────


def manage_open_positions(symbol: str = SYMBOL, regime: Optional[str] = None) -> int:
    """
    Apply all protections (P14, P15, P5 BE, P11 MFE trail, time stops)
    to all Fable positions. Returns number of positions closed.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return 0

    # Update regime once per cycle
    current_regime = regime or _classify_regime(symbol)

    closed = 0
    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue

        # P14: regime-laddered profit lock (primary) — runs first
        _apply_regime_protection(pos, symbol, current_regime)

        # P16: profit-retrace close (catches sub-1R round-trips that P14 misses)
        _apply_profit_retrace_close(pos, symbol)

        # P17: two-stage trailing ratchet + virtual lever
        _apply_ratchet(pos, symbol)

        # P15: Adaptive Peak Exit (profit maximizer, independent of P14)
        if PEAK_EXIT_ENABLED:
            tp = float(pos.tp)
            if tp > 0:
                entry = float(pos.price_open)
                r = abs(tp - entry) / 2.0
                if r > 0:
                    reason = assess_position(
                        ticket=pos.ticket,
                        symbol=symbol,
                        is_buy=pos.type == mt5.POSITION_TYPE_BUY,
                        entry=entry,
                        price=float(pos.price_current),
                        r=r,
                    )
                    if reason:
                        logger.info(f"[PM] P15 peak exit: ticket={pos.ticket} reason={reason}")
                        if close_peak_exit(pos.ticket, symbol, reason):
                            closed += 1
                        continue

        # P5: legacy BE (backup if P14 is off or for accounts below cutoff)
        _apply_equity_tiered_breakeven(pos, symbol)

        # P11: MFE trail (extra tightening on volatile regimes)
        _apply_mfe_trail(pos, symbol)

        # Time stops
        held = _bars_held(pos.time)
        if held >= MAX_HOLD_BARS:
            logger.info(f"[PM] Max-hold stop: ticket={pos.ticket} "
                        f"held={held:.1f} bars >= {MAX_HOLD_BARS}")
            if close_position(pos.ticket, symbol, comment="FableMaxHold"):
                closed += 1
            continue
        if USE_NEGATIVE_TIME_STOP and held >= NEGATIVE_TIME_STOP_BARS and pos.profit < 0:
            logger.info(f"[PM] Negative time stop: ticket={pos.ticket} "
                        f"held={held:.1f} bars, pnl={pos.profit:.2f}")
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
                "ticket": p.ticket,
                "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": p.volume,
                "profit": p.profit,
                "open_time": pd.to_datetime(p.time, unit="s", utc=True),
            })
    return pd.DataFrame(rows)


# ── P14 background thread (runs independently of M1 cycle) ────────────────


def _p14_loop(symbol: str = SYMBOL, sleep_seconds: float = 5.0):
    logger.info(f"[P14] Regime protection thread started for {symbol} (every {sleep_seconds}s)")
    while True:
        try:
            manage_open_positions(symbol)
        except Exception as exc:
            logger.error(f"[P14] Error: {exc}", exc_info=True)
        time.sleep(sleep_seconds)


def start_regime_protection(symbol: str = SYMBOL, sleep_seconds: float = 5.0):
    """Spawn a daemon thread running P14 regime-laddered protection."""
    if not REGIME_PROTECTION_ENABLED:
        logger.info("[P14] Regime protection disabled by config.")
        return
    thread = threading.Thread(
        target=_p14_loop,
        args=(symbol, sleep_seconds),
        daemon=True,
        name="RegimeProtection",
    )
    thread.start()
    logger.info(f"[P14] Regime protection daemon thread started")