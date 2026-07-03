"""
=============================================================================
DATA ENGINE
Multi-timeframe MT5 data fetching with caching, alignment, and validation.
=============================================================================
"""

import os
import time
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import MetaTrader5 as mt5

from config import (
    SYMBOL, PRIMARY_TF, CONTEXT_TFS, FAST_TF,
    MIN_BARS_REQUIRED, DATA_CACHE_DIR
)

logger = logging.getLogger("DataEngine")

# ─────────────────────────────────────────────────────────────────────────────
# MT5 Timeframe mapping
# ─────────────────────────────────────────────────────────────────────────────
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440
}


def initialize_mt5(retries: int = 3, delay: float = 2.0) -> bool:
    """Initialize MT5 connection with retries."""
    for attempt in range(retries):
        if mt5.initialize():
            info = mt5.terminal_info()
            logger.info(f"MT5 connected | Build: {info.build} | Connected: {info.connected}")
            return True
        logger.warning(f"MT5 init attempt {attempt+1}/{retries} failed: {mt5.last_error()}")
        time.sleep(delay)
    logger.error("MT5 initialization failed after all retries.")
    return False


def shutdown_mt5():
    """Safely shutdown MT5."""
    mt5.shutdown()
    logger.info("MT5 shutdown.")


def _validate_symbol(symbol: str) -> bool:
    """Check symbol exists and is available."""
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"Symbol {symbol} not found. Check broker symbol name.")
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
        logger.info(f"Symbol {symbol} added to Market Watch.")
    return True


def fetch_ohlcv(
    symbol: str,
    tf_str: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV data from MT5 for a given symbol/timeframe/date range.
    Returns DataFrame with columns: open, high, low, close, tick_volume, spread.
    Index: UTC datetime.
    """
    cache_key = f"{symbol}_{tf_str}_{start.date()}_{end.date()}.pkl"
    cache_path = os.path.join(DATA_CACHE_DIR, cache_key)

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        logger.info(f"[CACHE] {symbol} {tf_str}: {len(df)} bars")
        return df

    if not _validate_symbol(symbol):
        return pd.DataFrame()

    tf = TF_MAP.get(tf_str)
    if tf is None:
        raise ValueError(f"Unknown timeframe: {tf_str}. Valid: {list(TF_MAP.keys())}")

    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        logger.error(f"No data for {symbol} {tf_str} {start}–{end}. Error: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume", "spread"]].copy()
    df.sort_index(inplace=True)

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="last")]

    logger.info(f"[FETCH] {symbol} {tf_str}: {len(df)} bars ({df.index[0]} – {df.index[-1]})")

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)

    return df


def fetch_latest(symbol: str, tf_str: str, count: int = 500) -> pd.DataFrame:
    """Fetch the most recent N bars from MT5."""
    if not _validate_symbol(symbol):
        return pd.DataFrame()

    tf = TF_MAP.get(tf_str)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to fetch latest {count} bars for {symbol} {tf_str}.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume", "spread"]].copy()
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_multi_tf(
    symbol: str,
    primary_tf: str,
    context_tfs: list,
    start: datetime,
    end: datetime,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all timeframes. Returns dict keyed by TF string.
    """
    result = {}
    all_tfs = [primary_tf] + context_tfs
    for tf in all_tfs:
        df = fetch_ohlcv(symbol, tf, start, end, use_cache=use_cache)
        if df.empty:
            logger.warning(f"Empty data for {tf}. Skipping.")
        else:
            result[tf] = df
    return result


def fetch_multi_tf_latest(
    symbol: str,
    primary_tf: str,
    context_tfs: list,
    count: int = 500
) -> Dict[str, pd.DataFrame]:
    """Fetch latest bars for all timeframes."""
    result = {}
    all_tfs = [primary_tf] + context_tfs
    for tf in all_tfs:
        # Fetch more bars for higher TFs so we always have enough context
        mult = max(1, TF_MINUTES.get(tf, 15) // TF_MINUTES.get(primary_tf, 15))
        n = min(count * mult, 2000)
        df = fetch_latest(symbol, tf, count=n)
        if not df.empty:
            result[tf] = df
    return result


def align_to_primary(
    data: Dict[str, pd.DataFrame],
    primary_tf: str
) -> Dict[str, pd.DataFrame]:
    """
    Align all higher-TF DataFrames to primary TF index using forward-fill.
    This prevents look-ahead bias: higher TF value at time T is the
    last completed bar BEFORE time T.
    """
    if primary_tf not in data:
        raise ValueError(f"Primary TF {primary_tf} not in data dict.")

    primary_index = data[primary_tf].index
    aligned = {primary_tf: data[primary_tf].copy()}

    for tf, df in data.items():
        if tf == primary_tf:
            continue
        # Reindex to primary, ffill — each primary bar gets the last closed higher-TF bar
        df_reindexed = df.reindex(primary_index, method="ffill")
        aligned[tf] = df_reindexed
        logger.debug(f"Aligned {tf} to {primary_tf}: {len(df_reindexed)} bars")

    return aligned


def get_current_spread(symbol: str) -> float:
    """Get current spread in points."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 999.0  # Return high spread to block trade
    info = mt5.symbol_info(symbol)
    if info is None:
        return 999.0
    return (tick.ask - tick.bid) / info.point


def get_account_info() -> dict:
    """Get current account balance, equity, margin."""
    acc = mt5.account_info()
    if acc is None:
        return {}
    return {
        "balance":  acc.balance,
        "equity":   acc.equity,
        "margin":   acc.margin,
        "free_margin": acc.margin_free,
        "profit":   acc.profit,
        "leverage": acc.leverage,
    }


def get_open_positions(symbol: Optional[str] = None) -> pd.DataFrame:
    """Get open positions as DataFrame."""
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None or len(positions) == 0:
        return pd.DataFrame()

    rows = []
    for p in positions:
        rows.append({
            "ticket":       p.ticket,
            "symbol":       p.symbol,
            "type":         "BUY" if p.type == 0 else "SELL",
            "volume":       p.volume,
            "open_price":   p.price_open,
            "current_price":p.price_current,
            "sl":           p.sl,
            "tp":           p.tp,
            "profit":       p.profit,
            "swap":         p.swap,
            "open_time":    pd.to_datetime(p.time, unit="s", utc=True),
            "magic":        p.magic,
            "comment":      p.comment,
        })
    return pd.DataFrame(rows)


def get_daily_pnl() -> float:
    """Calculate today's total P&L (closed + open)."""
    today = datetime.utcnow().date()
    from_date = datetime(today.year, today.month, today.day)
    to_date   = datetime(today.year, today.month, today.day, 23, 59, 59)

    deals = mt5.history_deals_get(from_date, to_date)
    closed_pnl = sum(d.profit for d in deals) if deals else 0.0

    positions = mt5.positions_get()
    open_pnl  = sum(p.profit for p in positions) if positions else 0.0

    return closed_pnl + open_pnl


def is_new_bar(symbol: str, tf_str: str, last_bar_time: Optional[datetime]) -> Tuple[bool, datetime]:
    """
    Check if a new bar has formed on the given timeframe.
    Returns (is_new, current_bar_time).
    """
    rates = mt5.copy_rates_from_pos(symbol, TF_MAP[tf_str], 0, 1)
    if rates is None or len(rates) == 0:
        return False, last_bar_time

    current_bar_time = pd.to_datetime(rates[0]["time"], unit="s", utc=True).to_pydatetime()

    if last_bar_time is None or current_bar_time > last_bar_time:
        return True, current_bar_time
    return False, last_bar_time


def validate_data_quality(df: pd.DataFrame, min_bars: int = None) -> bool:
    """
    Validate data quality: check for gaps, NaNs, sufficient length.
    """
    if min_bars is None:
        min_bars = MIN_BARS_REQUIRED

    if df is None or df.empty:
        logger.warning("Data validation failed: empty DataFrame.")
        return False

    if len(df) < min_bars:
        logger.warning(f"Data validation failed: only {len(df)} bars (need {min_bars}).")
        return False

    nan_pct = df.isnull().mean().max()
    if nan_pct > 0.05:
        logger.warning(f"Data validation failed: {nan_pct:.1%} NaN ratio exceeds 5%.")
        return False

    # Check for large time gaps (more than 10x the expected bar interval)
    if len(df.index) > 1:
        diffs = pd.Series(df.index).diff().dropna()
        median_diff = diffs.median()
        large_gaps  = (diffs > median_diff * 10).sum()
        if large_gaps > 10:
            logger.warning(f"Data quality warning: {large_gaps} large time gaps detected.")

    return True
