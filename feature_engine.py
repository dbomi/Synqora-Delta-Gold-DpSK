"""
=============================================================================
FEATURE ENGINE
50+ features: returns, volatility, structure, momentum, session, MTF context.
Leading-biased design — no simple lagging indicator copies.
=============================================================================
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

from config import (
    ATR_PERIOD, RSI_PERIOD, FAST_EMA, SLOW_EMA, TREND_EMA,
    VWAP_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    MOMENTUM_PERIODS, VOLATILITY_PERIODS, SESSIONS,
    PRIMARY_TF, CONTEXT_TFS
)

logger = logging.getLogger("FeatureEngine")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _rolling_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).std()


def _rank(series: pd.Series, period: int) -> pd.Series:
    """Percentile rank of current value over rolling window [0, 1]."""
    return series.rolling(period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE FEATURE BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """ATR-normalized returns at multiple horizons."""
    atr = _atr(df, ATR_PERIOD)
    df["atr"] = atr
    df["atr_pct"] = atr / df["close"]

    for p in MOMENTUM_PERIODS:
        ret = df["close"].pct_change(p)
        df[f"ret_{p}"]       = ret
        df[f"ret_{p}_norm"]  = ret / (atr / df["close"] + 1e-9)  # ATR-normalized

    # Bar-level features
    df["bar_range"]     = (df["high"] - df["low"]) / (atr + 1e-9)
    df["bar_body"]      = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-9)
    df["bar_direction"] = np.sign(df["close"] - df["open"])

    # Upper and lower wick
    body_top    = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    df["upper_wick"] = (df["high"] - body_top)    / (df["high"] - df["low"] + 1e-9)
    df["lower_wick"] = (body_bottom - df["low"])  / (df["high"] - df["low"] + 1e-9)

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum acceleration, EMA alignment, RSI velocity."""
    df["ema_fast"]  = _ema(df["close"], FAST_EMA)
    df["ema_slow"]  = _ema(df["close"], SLOW_EMA)
    df["ema_trend"] = _ema(df["close"], TREND_EMA)

    # EMA spread (normalized by ATR)
    df["ema_spread"]       = (df["ema_fast"] - df["ema_slow"]) / (df["atr"] + 1e-9)
    df["ema_alignment"]    = np.where(
        (df["ema_fast"] > df["ema_slow"]) & (df["ema_slow"] > df["ema_trend"]), 1,
        np.where(
            (df["ema_fast"] < df["ema_slow"]) & (df["ema_slow"] < df["ema_trend"]), -1, 0
        )
    )
    df["price_vs_trend"]   = (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-9)

    # RSI and RSI velocity
    rsi = _rsi(df["close"], RSI_PERIOD)
    df["rsi"]       = rsi
    df["rsi_vel"]   = rsi.diff(3)
    df["rsi_norm"]  = (rsi - 50) / 50  # Centered [-1, 1]

    # Momentum acceleration (second derivative)
    for p in [5, 10]:
        mom  = df["close"].pct_change(p)
        df[f"mom_accel_{p}"] = mom.diff(p)

    # MACD (but normalized)
    macd       = _ema(df["close"], 12) - _ema(df["close"], 26)
    macd_sig   = _ema(macd, 9)
    df["macd_norm"]    = macd / (df["atr"] + 1e-9)
    df["macd_hist"]    = (macd - macd_sig) / (df["atr"] + 1e-9)

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility state: realized vol, compression, expansion."""
    for p in VOLATILITY_PERIODS:
        rv = df["close"].pct_change().rolling(p).std() * np.sqrt(p)
        df[f"realized_vol_{p}"] = rv

    # Bollinger Bands
    bb_mid  = df["close"].rolling(BOLLINGER_PERIOD).mean()
    bb_std  = df["close"].rolling(BOLLINGER_PERIOD).std()
    bb_up   = bb_mid + BOLLINGER_STD * bb_std
    bb_low  = bb_mid - BOLLINGER_STD * bb_std
    bb_wid  = (bb_up - bb_low) / (bb_mid + 1e-9)

    df["bb_position"]  = (df["close"] - bb_low) / (bb_up - bb_low + 1e-9)  # [0,1]
    df["bb_width"]     = bb_wid
    df["bb_width_rank"]= bb_wid.rolling(60).rank(pct=True)

    # ATR ratio (current ATR vs longer-term ATR) — compression signal
    atr_fast = _atr(df, 7)
    atr_slow = _atr(df, 28)
    df["atr_ratio"]    = atr_fast / (atr_slow + 1e-9)
    df["vol_compress"] = (df["atr_ratio"] < 0.7).astype(int)
    df["vol_expand"]   = (df["atr_ratio"] > 1.3).astype(int)

    # Historical volatility percentile
    hv20 = df["close"].pct_change().rolling(20).std()
    df["hv_percentile"] = hv20.rolling(252).rank(pct=True)

    return df


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market structure: swing highs/lows, BOS, CHoCH, VWAP distance,
    rolling max/min distances.
    """
    # Rolling highs and lows (structure levels)
    for p in [10, 20, 40]:
        roll_high = df["high"].rolling(p).max()
        roll_low  = df["low"].rolling(p).min()
        df[f"dist_high_{p}"] = (roll_high - df["close"]) / (df["atr"] + 1e-9)
        df[f"dist_low_{p}"]  = (df["close"] - roll_low)  / (df["atr"] + 1e-9)
        # Position within range
        df[f"pos_in_range_{p}"] = (df["close"] - roll_low) / (roll_high - roll_low + 1e-9)

    # Break of structure (simple: new N-bar high/low after opposite move)
    df["new_high_20"] = (df["high"] > df["high"].shift(1).rolling(19).max()).astype(int)
    df["new_low_20"]  = (df["low"]  < df["low"].shift(1).rolling(19).min()).astype(int)

    # Rolling VWAP (volume-weighted average price over N bars)
    typical_price  = (df["high"] + df["low"] + df["close"]) / 3
    tv             = typical_price * df["tick_volume"]
    vwap           = tv.rolling(VWAP_PERIOD).sum() / (df["tick_volume"].rolling(VWAP_PERIOD).sum() + 1e-9)
    df["vwap_dist"] = (df["close"] - vwap) / (df["atr"] + 1e-9)  # +ve = above VWAP

    # Consecutive directional bars
    direction = np.sign(df["close"] - df["open"])
    df["consec_bull"] = direction.groupby(
        (direction != direction.shift()).cumsum()
    ).cumcount().where(direction == 1, 0)
    df["consec_bear"] = direction.groupby(
        (direction != direction.shift()).cumsum()
    ).cumcount().where(direction == -1, 0)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tick volume features: burst, imbalance proxy, relative volume."""
    vol = df["tick_volume"].astype(float)

    df["vol_ma"]       = vol.rolling(20).mean()
    df["rel_volume"]   = vol / (df["vol_ma"] + 1e-9)
    df["vol_burst"]    = (df["rel_volume"] > 2.0).astype(int)

    # Volume-price relationship (effort vs result)
    ret_abs = df["close"].pct_change().abs()
    df["vol_price_ratio"] = vol / (ret_abs * 10000 + 1e-9)  # high = inefficient move

    # Volume rank
    df["vol_rank"] = vol.rolling(50).rank(pct=True)

    # Spread features (higher spread = lower liquidity)
    if "spread" in df.columns:
        df["spread_norm"]  = df["spread"] / (df["atr"] * 10000 / df["close"] + 1e-9)
        df["spread_rank"]  = df["spread"].rolling(50).rank(pct=True)

    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time and session encoding — Gold has strong session-based behavior."""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    hour = df.index.hour
    dow  = df.index.dayofweek  # Monday=0

    # Cyclical time encoding (prevents discontinuity at midnight)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow  / 5)
    df["dow_cos"]  = np.cos(2 * np.pi * dow  / 5)

    # Session flags
    for sess_name, (start_h, end_h) in SESSIONS.items():
        if start_h < end_h:
            mask = (hour >= start_h) & (hour < end_h)
        else:  # wraps midnight
            mask = (hour >= start_h) | (hour < end_h)
        df[f"sess_{sess_name.lower()}"] = mask.astype(int)

    # London-NY overlap (highest Gold volume period)
    overlap_mask = (hour >= 12) & (hour < 16)
    df["sess_overlap"] = overlap_mask.astype(int)

    # Time since London open (mean reversion anchor)
    london_open_hour = 7
    hours_since_london = (hour - london_open_hour) % 24
    df["hrs_since_london"] = hours_since_london

    return df


def add_mtf_features(
    df_primary: pd.DataFrame,
    df_context: Dict[str, pd.DataFrame],
    primary_tf: str = PRIMARY_TF
) -> pd.DataFrame:
    """
    Add higher-timeframe context features to primary TF DataFrame.
    All context values are already aligned (forward-filled) from data_engine.
    """
    for tf, df_ctx in df_context.items():
        if tf == primary_tf or df_ctx.empty:
            continue

        # ATR-normalized return on context TF
        ctx_atr = _atr(df_ctx, ATR_PERIOD)

        # Trend alignment: is context TF bullish or bearish?
        ctx_ema_fast = _ema(df_ctx["close"], FAST_EMA)
        ctx_ema_slow = _ema(df_ctx["close"], SLOW_EMA)
        ctx_trend    = np.sign(ctx_ema_fast - ctx_ema_slow)

        # Context RSI
        ctx_rsi   = _rsi(df_ctx["close"], RSI_PERIOD)

        # Position vs context range
        ctx_high20 = df_ctx["high"].rolling(20).max()
        ctx_low20  = df_ctx["low"].rolling(20).min()
        ctx_pos    = (df_ctx["close"] - ctx_low20) / (ctx_high20 - ctx_low20 + 1e-9)

        # Reindex to primary
        prefix = tf.lower()
        df_primary[f"{prefix}_trend"]    = ctx_trend.reindex(df_primary.index, method="ffill")
        df_primary[f"{prefix}_rsi"]      = ctx_rsi.reindex(df_primary.index, method="ffill")
        df_primary[f"{prefix}_pos"]      = ctx_pos.reindex(df_primary.index, method="ffill")
        df_primary[f"{prefix}_atr_norm"] = (ctx_atr / df_ctx["close"]).reindex(df_primary.index, method="ffill")

    return df_primary


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    data: Dict[str, pd.DataFrame],
    primary_tf: str = PRIMARY_TF
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        data: Dict of aligned DataFrames keyed by TF string.
        primary_tf: The primary timeframe key.

    Returns:
        DataFrame with all features, indexed to primary TF.
    """
    if primary_tf not in data or data[primary_tf].empty:
        raise ValueError(f"Primary TF {primary_tf} not found in data dict.")

    df = data[primary_tf].copy()

    # Feature building can run frequently during live virtual-entry polling.
    # Keep this at DEBUG so trade/reason logs remain readable.
    logger.debug(f"Building features on {len(df)} bars...")

    # Core blocks
    df = add_return_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_structure_features(df)
    df = add_volume_features(df)
    df = add_session_features(df)

    # Higher-TF context
    ctx = {k: v for k, v in data.items() if k != primary_tf}
    if ctx:
        df = add_mtf_features(df, ctx, primary_tf)

    # Drop rows with NaNs from indicator warmup
    initial_len = len(df)
    df.dropna(inplace=True)
    logger.debug(f"Feature build complete: {len(df)} bars ({initial_len - len(df)} dropped for warmup)")

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return only engineered feature columns (exclude OHLCV source columns).
    """
    exclude = {"open", "high", "low", "close", "tick_volume", "spread", "atr"}
    return [c for c in df.columns if c not in exclude]


def build_live_features(
    data: Dict[str, pd.DataFrame],
    training_columns: List[str],
    primary_tf: str = PRIMARY_TF
) -> Optional[pd.DataFrame]:
    """
    Build features for live trading and ensure column alignment with training.
    """
    try:
        df = build_features(data, primary_tf)
        if df.empty:
            return None

        feat_cols = get_feature_columns(df)
        available = [c for c in training_columns if c in feat_cols]
        missing   = [c for c in training_columns if c not in feat_cols]

        if missing:
            logger.warning(f"Live feature build: {len(missing)} columns missing from training set.")
            for col in missing:
                df[col] = 0.0  # Impute with zero — will be caught by uncertainty gate

        return df[training_columns]

    except Exception as e:
        logger.error(f"Live feature build failed: {e}", exc_info=True)
        return None
