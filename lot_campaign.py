"""
=============================================================================
LOT CAMPAIGN — SYNQORA DELTA GOLD FABLE
Scale-campaign position sizing: risk-percent compounding with A+ dual
entry. Pure math lives in compute_signal_lots() (unit-testable without
MT5); get_signal_lots() is the live wrapper that reads account state.

Sizing:  lot = equity × RISK_PCT / (SL distance in $ per lot)
         → risk at 1R is a constant fraction of equity, so lots grow
           from 0.02-scale at $500 toward CAMPAIGN_MAX_LOT as equity grows.
A+:      signals with model prob >= A_PLUS_PROB_THRESHOLD open
         A_PLUS_POSITION_COUNT tickets (combined <= MAX_TOTAL_SIGNAL_LOT).
Safety:  rolling drawdown degrade multiplier, margin headroom cap,
         broker volume bounds.
=============================================================================
"""

import logging
import math
from typing import Tuple

from config import (
    SYMBOL, LOT_SIZE, LOT_SIZING_MODE, RISK_PCT_PER_TRADE,
    CAMPAIGN_MAX_LOT, MAX_TOTAL_SIGNAL_LOT, CONTRACT_USD_PER_UNIT,
    MARGIN_USE_CAP, A_PLUS_DUAL_ENTRY, A_PLUS_PROB_THRESHOLD,
    A_PLUS_POSITION_COUNT, A_PLUS_MIN_EQUITY, TRIPLE_BARRIER_SL_ATR,
    GOLDEN_HOUR_CAMPAIGN_ENABLED, GOLDEN_HOURS_UTC, GOLDEN_HOUR_LOT_MULT,
    GOLDEN_HOUR_A_PLUS_THRESHOLD, DEAD_HOURS_UTC, DEAD_HOUR_LOT_MULT,
)

logger = logging.getLogger("LotCampaign")


def _hour_multiplier() -> float:
    """Return sizing multiplier based on current UTC hour."""
    if not GOLDEN_HOUR_CAMPAIGN_ENABLED:
        return 1.0
    from datetime import datetime, timezone
    h = datetime.now(timezone.utc).hour
    if h in GOLDEN_HOURS_UTC:
        return GOLDEN_HOUR_LOT_MULT
    if h in DEAD_HOURS_UTC:
        return DEAD_HOUR_LOT_MULT
    return 1.0


def _is_golden_hour() -> bool:
    if not GOLDEN_HOUR_CAMPAIGN_ENABLED:
        return False
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).hour in GOLDEN_HOURS_UTC


def compute_signal_lots(
    equity:        float,
    m15_atr:       float,
    prob:          float,
    dd_mult:       float = 1.0,
    margin_room_lots: float = float("inf"),
    volume_step:   float = 0.01,
    volume_min:    float = 0.01,
    hour_mult:     float = 1.0,
    a_plus_threshold: float | None = None,
) -> Tuple[float, int]:
    """
    Pure sizing math. Returns (lot_per_position, n_positions).
    (0.0, 0) means the signal cannot be sized (no equity / no margin room).
    `hour_mult` scales the base lot (from golden/dead hour campaign).
    `a_plus_threshold` overrides A_PLUS_PROB_THRESHOLD if provided.
    """
    threshold = a_plus_threshold if a_plus_threshold is not None else A_PLUS_PROB_THRESHOLD
    n = A_PLUS_POSITION_COUNT if (
        A_PLUS_DUAL_ENTRY
        and prob >= threshold
        and equity >= A_PLUS_MIN_EQUITY
    ) else 1

    if LOT_SIZING_MODE == "RISK_PCT":
        if equity <= 0 or m15_atr <= 0:
            return 0.0, 0
        r_usd_per_lot = TRIPLE_BARRIER_SL_ATR * m15_atr * CONTRACT_USD_PER_UNIT
        base = equity * RISK_PCT_PER_TRADE / 100.0 / max(r_usd_per_lot, 1e-9)
    elif LOT_SIZING_MODE == "DELTA":
        from dynamic_lot_sizer import raw_dynamic_lot
        base = raw_dynamic_lot(equity)
    else:
        base = LOT_SIZE

    base = min(base * dd_mult * hour_mult, CAMPAIGN_MAX_LOT)
    total = min(base * n, MAX_TOTAL_SIGNAL_LOT, margin_room_lots)

    lot_each = math.floor(total / n / volume_step + 1e-9) * volume_step
    if lot_each < volume_min:
        # Not enough for n positions — try a single one.
        n = 1
        lot_each = math.floor(min(total, base) / volume_step + 1e-9) * volume_step
        if lot_each < volume_min:
            if margin_room_lots >= volume_min:
                lot_each = volume_min
                n = 1
            else:
                return 0.0, 0
    return round(lot_each, 2), n


def get_signal_lots(side: str, prob: float, m15_atr: float,
                    streak_mult: float = 1.0) -> Tuple[float, int]:
    """
    Live wrapper: reads account equity, drawdown state, margin headroom and
    broker volume bounds from MT5, then delegates to compute_signal_lots.

    P1: `streak_mult` is a multiplier from MetaAgent.streak_multiplier() based
    on consecutive win/loss streaks.
    """
    import MetaTrader5 as mt5
    from dynamic_lot_sizer import drawdown_multiplier, get_account_equity_balance

    equity, balance = get_account_equity_balance()
    eq = equity or balance
    if eq <= 0:
        logger.warning("[CAMPAIGN] No account equity reading; sizing 0.")
        return 0.0, 0
    dd_mult = drawdown_multiplier(eq) * streak_mult

    info = mt5.symbol_info(SYMBOL)
    step = float(getattr(info, "volume_step", 0.01) or 0.01) if info else 0.01
    vmin = float(getattr(info, "volume_min", 0.01) or 0.01) if info else 0.01

    # Margin headroom: how many lots fit in MARGIN_USE_CAP of free margin?
    margin_room = float("inf")
    tick = mt5.symbol_info_tick(SYMBOL)
    acc = mt5.account_info()
    if tick is not None and acc is not None:
        order_type = mt5.ORDER_TYPE_BUY if str(side).upper() == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        margin_1lot = mt5.order_calc_margin(order_type, SYMBOL, 1.0, price)
        if margin_1lot and margin_1lot > 0:
            margin_room = max(0.0, acc.margin_free * MARGIN_USE_CAP / margin_1lot)

    hm = _hour_multiplier()
    a_plus_t = GOLDEN_HOUR_A_PLUS_THRESHOLD if _is_golden_hour() else None

    lot_each, n = compute_signal_lots(
        equity=eq, m15_atr=m15_atr, prob=prob, dd_mult=dd_mult,
        margin_room_lots=margin_room, volume_step=step, volume_min=vmin,
        hour_mult=hm, a_plus_threshold=a_plus_t,
    )
    if n > 0:
        logger.info(f"[CAMPAIGN] {side} prob={prob:.2f} equity={eq:.2f} "
                    f"dd_mult={dd_mult:.2f} -> {n} x {lot_each} lots")
    return lot_each, n
