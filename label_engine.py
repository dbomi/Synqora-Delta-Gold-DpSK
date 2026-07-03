"""
=============================================================================
LABEL ENGINE
Professional trade labeling: triple-barrier method, MFE/MAE tracking,
+1R before -1R probability. Replaces naive "next close" prediction target.
=============================================================================
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

from config import (
    TRIPLE_BARRIER_TP_ATR, TRIPLE_BARRIER_SL_ATR,
    TRIPLE_BARRIER_MAX_BARS, MFE_LOOKFORWARD, ATR_PERIOD,
    MIN_R_RATIO
)

logger = logging.getLogger("LabelEngine")


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """ATR via EWM."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRIPLE BARRIER LABELING
# ─────────────────────────────────────────────────────────────────────────────

def triple_barrier_labels(
    df: pd.DataFrame,
    tp_atr_mult: float = TRIPLE_BARRIER_TP_ATR,
    sl_atr_mult: float = TRIPLE_BARRIER_SL_ATR,
    max_bars:    int   = TRIPLE_BARRIER_MAX_BARS,
    side:        str   = "both"   # "buy", "sell", "both"
) -> pd.DataFrame:
    """
    Triple barrier labeling for BUY and/or SELL specialist models.

    Labels:
        BUY side:
            +1 = price reached TP (+tp_atr) before SL (-sl_atr) within max_bars
             0 = timed out (neither barrier hit)
            -1 = price hit SL before TP (loss)

        SELL side:
            +1 = price reached TP (-tp_atr) before SL (+sl_atr) within max_bars
             0 = timeout
            -1 = hit SL (loss)

    Returns original df with added columns:
        atr_at_entry, buy_tp, buy_sl, sell_tp, sell_sl,
        buy_label, sell_label, buy_bars_held, sell_bars_held
    """
    atr = _compute_atr(df, ATR_PERIOD)
    prices = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = atr.values
    n      = len(df)

    buy_labels   = np.zeros(n, dtype=np.int8)
    sell_labels  = np.zeros(n, dtype=np.int8)
    buy_bars     = np.full(n, max_bars, dtype=np.int16)
    sell_bars    = np.full(n, max_bars, dtype=np.int16)
    buy_tp_arr   = np.zeros(n)
    buy_sl_arr   = np.zeros(n)
    sell_tp_arr  = np.zeros(n)
    sell_sl_arr  = np.zeros(n)

    for i in range(n - 1):
        entry     = prices[i]
        atr_i     = atrs[i]
        buy_tp_p  = entry + tp_atr_mult * atr_i
        buy_sl_p  = entry - sl_atr_mult * atr_i
        sell_tp_p = entry - tp_atr_mult * atr_i
        sell_sl_p = entry + sl_atr_mult * atr_i

        buy_tp_arr[i]  = buy_tp_p
        buy_sl_arr[i]  = buy_sl_p
        sell_tp_arr[i] = sell_tp_p
        sell_sl_arr[i] = sell_sl_p

        # Look forward up to max_bars
        for j in range(i + 1, min(i + max_bars + 1, n)):
            bar_high = highs[j]
            bar_low  = lows[j]
            elapsed  = j - i

            if side in ("buy", "both"):
                if buy_labels[i] == 0:
                    if bar_high >= buy_tp_p:
                        buy_labels[i] = 1
                        buy_bars[i]   = elapsed
                    elif bar_low <= buy_sl_p:
                        buy_labels[i] = -1
                        buy_bars[i]   = elapsed

            if side in ("sell", "both"):
                if sell_labels[i] == 0:
                    if bar_low <= sell_tp_p:
                        sell_labels[i] = 1
                        sell_bars[i]   = elapsed
                    elif bar_high >= sell_sl_p:
                        sell_labels[i] = -1
                        sell_bars[i]   = elapsed

            if buy_labels[i] != 0 and sell_labels[i] != 0:
                break

    result = df.copy()
    result["atr_at_entry"]  = atr
    result["buy_tp"]        = buy_tp_arr
    result["buy_sl"]        = buy_sl_arr
    result["sell_tp"]       = sell_tp_arr
    result["sell_sl"]       = sell_sl_arr
    result["buy_label"]     = buy_labels
    result["sell_label"]    = sell_labels
    result["buy_bars_held"] = buy_bars
    result["sell_bars_held"]= sell_bars

    # Binary win/loss labels (0=loss/timeout, 1=win)
    result["buy_win"]  = (result["buy_label"]  == 1).astype(int)
    result["sell_win"] = (result["sell_label"] == 1).astype(int)

    _log_label_stats(result)
    return result


def _log_label_stats(df: pd.DataFrame):
    """Log label distribution for quality check."""
    for side in ["buy", "sell"]:
        col = f"{side}_label"
        if col not in df.columns:
            continue
        total = len(df)
        win   = (df[col] == 1).sum()
        loss  = (df[col] == -1).sum()
        tout  = (df[col] == 0).sum()
        logger.info(
            f"[Labels] {side.upper()}: "
            f"WIN={win}({win/total:.1%}) "
            f"LOSS={loss}({loss/total:.1%}) "
            f"TIMEOUT={tout}({tout/total:.1%})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MFE / MAE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfe_mae(
    df: pd.DataFrame,
    lookforward: int = MFE_LOOKFORWARD,
    side: str = "both"
) -> pd.DataFrame:
    """
    Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    over the next N bars after entry.

    MFE: best unrealized profit achievable during trade.
    MAE: worst unrealized loss experienced during trade.

    Both normalized by ATR at entry.
    """
    prices = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr_at_entry"].values if "atr_at_entry" in df.columns else (
        _compute_atr(df, ATR_PERIOD).values
    )
    n = len(df)

    buy_mfe  = np.zeros(n)
    buy_mae  = np.zeros(n)
    sell_mfe = np.zeros(n)
    sell_mae = np.zeros(n)

    for i in range(n - 1):
        end     = min(i + lookforward + 1, n)
        fwd_hi  = highs[i+1:end]
        fwd_lo  = lows[i+1:end]
        entry   = prices[i]
        atr_i   = atrs[i] + 1e-9

        if side in ("buy", "both"):
            buy_mfe[i] = (fwd_hi.max() - entry) / atr_i if len(fwd_hi) else 0.0
            buy_mae[i] = (entry - fwd_lo.min()) / atr_i if len(fwd_lo) else 0.0

        if side in ("sell", "both"):
            sell_mfe[i] = (entry - fwd_lo.min()) / atr_i if len(fwd_lo) else 0.0
            sell_mae[i] = (fwd_hi.max() - entry) / atr_i if len(fwd_hi) else 0.0

    result = df.copy()
    result["buy_mfe"]   = buy_mfe
    result["buy_mae"]   = buy_mae
    result["sell_mfe"]  = sell_mfe
    result["sell_mae"]  = sell_mae

    # MFE/MAE ratio — quality of trade (>1 means trade went your way more than against)
    result["buy_mfe_mae_ratio"]  = buy_mfe  / (buy_mae  + 1e-9)
    result["sell_mfe_mae_ratio"] = sell_mfe / (sell_mae + 1e-9)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# R-PROBABILITY LABEL
# ─────────────────────────────────────────────────────────────────────────────

def r_probability_label(
    df: pd.DataFrame,
    tp_atr_mult: float = TRIPLE_BARRIER_TP_ATR,
    sl_atr_mult: float = TRIPLE_BARRIER_SL_ATR,
    max_bars:    int   = TRIPLE_BARRIER_MAX_BARS
) -> pd.DataFrame:
    """
    Compute the probability of hitting +1R before -1R target.

    This is the primary prediction target: "Will this trade reach TP before SL?"
    Combined with triple_barrier_labels for training.
    """
    # Already computed via triple_barrier_labels — buy_win and sell_win are the +1R labels
    if "buy_win" not in df.columns:
        df = triple_barrier_labels(df, tp_atr_mult, sl_atr_mult, max_bars)

    # Rolling win rate as a soft probability estimate (useful for threshold calibration)
    window = 100
    df["buy_win_rate_local"]  = df["buy_win"].rolling(window, min_periods=20).mean()
    df["sell_win_rate_local"] = df["sell_win"].rolling(window, min_periods=20).mean()

    # Expected R (win% * R_ratio - loss% * 1)
    # Uses the ratio: TP / SL = tp_atr_mult / sl_atr_mult
    r_ratio = tp_atr_mult / sl_atr_mult
    win_rate_buy  = df["buy_win_rate_local"].fillna(0.5)
    win_rate_sell = df["sell_win_rate_local"].fillna(0.5)

    df["buy_expected_r"]  = win_rate_buy  * r_ratio - (1 - win_rate_buy)
    df["sell_expected_r"] = win_rate_sell * r_ratio - (1 - win_rate_sell)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FULL LABEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_labels(
    df: pd.DataFrame,
    tp_atr_mult: float = TRIPLE_BARRIER_TP_ATR,
    sl_atr_mult: float = TRIPLE_BARRIER_SL_ATR,
    max_bars:    int   = TRIPLE_BARRIER_MAX_BARS,
    compute_mfe: bool  = True
) -> pd.DataFrame:
    """
    Full label pipeline:
    1. Triple-barrier labels (buy_win, sell_win, buy_label, sell_label)
    2. MFE/MAE tracking (optional, adds computation time)
    3. R-probability context labels

    Returns df with all label columns added.
    The primary training targets are:
        buy_win  (1 = BUY trade wins, 0 = BUY trade loses/times out)
        sell_win (1 = SELL trade wins, 0 = SELL trade loses/times out)
    """
    logger.info(f"Building labels: TP={tp_atr_mult}×ATR, SL={sl_atr_mult}×ATR, max_bars={max_bars}")

    # Step 1: Triple barrier
    df = triple_barrier_labels(df, tp_atr_mult, sl_atr_mult, max_bars)

    # Step 2: MFE/MAE
    if compute_mfe:
        df = compute_mfe_mae(df)

    # Step 3: R-probability context
    df = r_probability_label(df, tp_atr_mult, sl_atr_mult, max_bars)

    # Drop last max_bars rows (incomplete labels)
    df = df.iloc[:-max_bars].copy()
    logger.info(f"Label build complete: {len(df)} labeled bars.")

    return df


def get_balanced_sample(
    df: pd.DataFrame,
    target: str = "buy_win",
    method: str = "undersample"
) -> pd.DataFrame:
    """
    Balance class distribution for training.
    method: "undersample" | "oversample" | "none"
    """
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]

    if method == "undersample":
        n = min(len(pos), len(neg))
        pos = pos.sample(n, random_state=42)
        neg = neg.sample(n, random_state=42)
    elif method == "oversample":
        n = max(len(pos), len(neg))
        pos = pos.sample(n, replace=True,  random_state=42)
        neg = neg.sample(n, replace=True,  random_state=42)

    balanced = pd.concat([pos, neg]).sort_index()
    logger.info(f"Balanced {target}: {len(pos)} pos, {len(neg)} neg → {len(balanced)} total")
    return balanced
