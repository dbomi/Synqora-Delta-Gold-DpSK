"""
=============================================================================
EXECUTION ENGINE — SYNQORA DELTA GOLD FABLE
Market-order execution only. Queue release IS the confirmation, so no
pending/limit routing is needed. Every trade enters with a hard SL and TP
derived from the validated label geometry (TP=2.0×ATR, SL=1.0×ATR).
=============================================================================
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5

from config import (
    SYMBOL, MAGIC_NUMBER, COMMENT, DEVIATION,
    MAX_SPREAD_POINTS, ORDER_RETRY_COUNT, ORDER_RETRY_DELAY,
    TRIPLE_BARRIER_TP_ATR, TRIPLE_BARRIER_SL_ATR,
)

logger = logging.getLogger("ExecutionEngine")

SUCCESS_RETCODES = {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL}


@dataclass
class OrderResult:
    success:   bool
    ticket:    int = 0
    retcode:   int = 0
    volume:    float = 0.0
    price:     float = 0.0
    sl:        float = 0.0
    tp:        float = 0.0
    error_msg: str = ""


def get_current_spread_points(symbol: str = SYMBOL) -> float:
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return 999.0
    return round((tick.ask - tick.bid) / info.point, 1)


def is_spread_acceptable(symbol: str = SYMBOL, max_spread: float = MAX_SPREAD_POINTS) -> bool:
    spread = get_current_spread_points(symbol)
    if spread > max_spread:
        logger.warning(f"Spread too wide: {spread:.1f} pts > {max_spread} pts. Skipping trade.")
        return False
    return True


def compute_sl_tp(side: str, entry_price: float, atr: float,
                  symbol: str = SYMBOL) -> tuple[float, float]:
    """SL/TP from the validated label geometry, rounded to symbol digits."""
    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 2

    sl_dist = TRIPLE_BARRIER_SL_ATR * atr
    tp_dist = TRIPLE_BARRIER_TP_ATR * atr

    if str(side).upper() == "BUY":
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist
    return round(sl, digits), round(tp, digits)


def place_market_order(
    side:    str,
    volume:  float,
    atr:     float,
    symbol:  str = SYMBOL,
    comment: str = COMMENT,
) -> OrderResult:
    """
    Place a market order with hard SL/TP. Retries transient failures up to
    ORDER_RETRY_COUNT times.
    """
    side = str(side).upper()
    if side not in ("BUY", "SELL"):
        return OrderResult(success=False, error_msg=f"invalid side {side}")

    if not is_spread_acceptable(symbol):
        return OrderResult(success=False, error_msg="spread_too_wide")

    order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL

    for attempt in range(1, ORDER_RETRY_COUNT + 1):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, error_msg="no_tick")

        price = tick.ask if side == "BUY" else tick.bid
        sl, tp = compute_sl_tp(side, price, atr, symbol)

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(volume),
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    DEVIATION,
            "magic":        MAGIC_NUMBER,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[EXEC] order_send returned None: {mt5.last_error()}")
            time.sleep(ORDER_RETRY_DELAY)
            continue

        if result.retcode in SUCCESS_RETCODES:
            logger.info(f"[EXEC] {side} {volume} {symbol} @ {result.price:.2f} "
                        f"SL={sl:.2f} TP={tp:.2f} ticket={result.order}")
            return OrderResult(
                success=True, ticket=result.order, retcode=result.retcode,
                volume=float(volume), price=float(result.price), sl=sl, tp=tp,
            )

        # Filling-mode fallback for brokers that reject IOC.
        if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            request["type_filling"] = mt5.ORDER_FILLING_FOK
            result = mt5.order_send(request)
            if result is not None and result.retcode in SUCCESS_RETCODES:
                logger.info(f"[EXEC] {side} filled with FOK fallback ticket={result.order}")
                return OrderResult(
                    success=True, ticket=result.order, retcode=result.retcode,
                    volume=float(volume), price=float(result.price), sl=sl, tp=tp,
                )

        logger.warning(f"[EXEC] Attempt {attempt}/{ORDER_RETRY_COUNT} failed: "
                       f"retcode={result.retcode} comment={result.comment}")
        time.sleep(ORDER_RETRY_DELAY)

    return OrderResult(success=False, retcode=result.retcode if result else 0,
                       error_msg="all_retries_failed")


def modify_position_sl(ticket: int, new_sl: float, symbol: str = SYMBOL) -> bool:
    """Move a position's SL (TP untouched). Returns True on success."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning(f"[EXEC] SL modify failed: ticket {ticket} not found.")
        return False
    pos = positions[0]
    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 2

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   symbol,
        "position": ticket,
        "sl":       round(float(new_sl), digits),
        "tp":       pos.tp,
        "magic":    MAGIC_NUMBER,
    }
    result = mt5.order_send(request)
    if result is not None and result.retcode in SUCCESS_RETCODES:
        logger.info(f"[EXEC] SL moved: ticket={ticket} {pos.sl:.2f} -> {new_sl:.2f}")
        return True
    logger.warning(f"[EXEC] SL modify failed for {ticket}: "
                   f"retcode={result.retcode if result else 'None'}")
    return False


def close_position(ticket: int, symbol: str = SYMBOL,
                   comment: str = "FableClose") -> bool:
    """Close an open position by ticket at market."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning(f"[EXEC] Close failed: ticket {ticket} not found.")
        return False
    pos = positions[0]

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    if pos.type == mt5.POSITION_TYPE_BUY:
        order_type, price = mt5.ORDER_TYPE_SELL, tick.bid
    else:
        order_type, price = mt5.ORDER_TYPE_BUY, tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       pos.volume,
        "type":         order_type,
        "position":     ticket,
        "price":        price,
        "deviation":    DEVIATION,
        "magic":        MAGIC_NUMBER,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is not None and result.retcode in SUCCESS_RETCODES:
        logger.info(f"[EXEC] Closed position {ticket} ({comment}).")
        return True
    logger.warning(f"[EXEC] Close position {ticket} failed: "
                   f"retcode={result.retcode if result else 'None'}")
    return False
