"""
=============================================================================
VALIDATION ENGINE
Walk-forward validation with purged embargo, stress testing, and full metrics.
Run before deploying any new model version to confirm it generalises.
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from config import (
    WF_N_SPLITS, WF_TEST_SIZE_BARS, WF_EMBARGO_BARS,
    MIN_PROFIT_FACTOR, MIN_WIN_RATE, MAX_DRAWDOWN_PCT,
    MIN_SHARPE, STRESS_SPREAD_MULT, STRESS_SLIPPAGE_POINTS,
    TRIPLE_BARRIER_TP_ATR, TRIPLE_BARRIER_SL_ATR, ATR_PERIOD
)

logger = logging.getLogger("ValidationEngine")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trades(
    df:            pd.DataFrame,
    buy_probs:     np.ndarray,
    sell_probs:    np.ndarray,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.60,
    spread_pts:    float = 20.0,
    slippage_pts:  float = 5.0,
    lot_size:      float = 0.01,
    tp_atr_mult:   float = TRIPLE_BARRIER_TP_ATR,
    sl_atr_mult:   float = TRIPLE_BARRIER_SL_ATR,
    max_hold_bars: int   = 48,
    cooldown:      int   = 3
) -> pd.DataFrame:
    """
    Simulate trades from model probabilities on OHLCV data.
    Returns DataFrame of individual trade results.
    """
    # Compute ATR
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values

    # Point value for XAUUSD (typically $1 per 0.01 lot per pip)
    PIP_VALUE = 1.0  # USD per 0.01 lot per $1 move — adjust for broker

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)

    trades = []
    last_trade_bar = -cooldown

    for i in range(n - 1):
        if (i - last_trade_bar) < cooldown:
            continue

        atr_i        = atr[i]
        spread_cost  = spread_pts  * 0.01   # Convert points to price
        slip_cost    = slippage_pts * 0.01

        is_buy  = buy_probs[i]  >= buy_threshold
        is_sell = sell_probs[i] >= sell_threshold

        if is_buy and is_sell:
            is_sell = False  # Prioritize higher confidence
            if sell_probs[i] > buy_probs[i]:
                is_buy = False

        if not (is_buy or is_sell):
            continue

        direction  = 1 if is_buy else -1
        entry      = closes[i] + (spread_cost + slip_cost) * direction
        tp_price   = entry + tp_atr_mult * atr_i * direction
        sl_price   = entry - sl_atr_mult * atr_i * direction

        # Forward simulate
        outcome = "TIMEOUT"
        exit_price = closes[min(i + max_hold_bars, n-1)]
        exit_bar   = min(i + max_hold_bars, n-1)

        for j in range(i + 1, min(i + max_hold_bars + 1, n)):
            bar_high = highs[j]
            bar_low  = lows[j]

            if direction == 1:   # BUY
                if bar_high >= tp_price:
                    outcome    = "WIN"
                    exit_price = tp_price
                    exit_bar   = j
                    break
                if bar_low <= sl_price:
                    outcome    = "LOSS"
                    exit_price = sl_price
                    exit_bar   = j
                    break
            else:                # SELL
                if bar_low <= tp_price:
                    outcome    = "WIN"
                    exit_price = tp_price
                    exit_bar   = j
                    break
                if bar_high >= sl_price:
                    outcome    = "LOSS"
                    exit_price = sl_price
                    exit_bar   = j
                    break

        pnl = (exit_price - entry) * direction * lot_size * 100  # Rough USD estimate

        trades.append({
            "entry_bar":   i,
            "exit_bar":    exit_bar,
            "direction":   "BUY" if direction == 1 else "SELL",
            "entry_price": entry,
            "exit_price":  exit_price,
            "sl":          sl_price,
            "tp":          tp_price,
            "outcome":     outcome,
            "pnl":         pnl,
            "bars_held":   exit_bar - i,
            "atr_at_entry":atr_i,
            "buy_prob":    buy_probs[i],
            "sell_prob":   sell_probs[i],
        })

        last_trade_bar = i

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> Dict:
    """Comprehensive trade metrics from simulation results."""
    if trades.empty:
        return {"error": "No trades to analyze"}

    wins   = trades[trades["outcome"] == "WIN"]
    losses = trades[trades["outcome"] == "LOSS"]

    total_pnl     = trades["pnl"].sum()
    win_pnl       = wins["pnl"].sum()   if not wins.empty   else 0.0
    loss_pnl      = abs(losses["pnl"].sum()) if not losses.empty else 0.0
    profit_factor = win_pnl / (loss_pnl + 1e-9)
    win_rate      = len(wins) / (len(trades) + 1e-9)

    # Cumulative PnL series for drawdown
    cum_pnl  = trades["pnl"].cumsum()
    running_max  = cum_pnl.cummax()
    drawdown     = (cum_pnl - running_max)
    max_dd       = float(drawdown.min())
    max_dd_pct   = abs(max_dd) / (running_max.max() + 1e-9) if running_max.max() > 0 else 0.0

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(trades) > 1:
        daily_pnl  = trades.groupby("entry_bar")["pnl"].sum()
        sharpe     = (daily_pnl.mean() / (daily_pnl.std() + 1e-9)) * np.sqrt(252 / 20)
    else:
        sharpe = 0.0

    # Expectancy (per trade)
    avg_win  = wins["pnl"].mean()   if not wins.empty   else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # CVaR (5th percentile of PnL distribution)
    cvar_5 = float(trades["pnl"].quantile(0.05))

    # BUY vs SELL breakdown
    buy_trades  = trades[trades["direction"] == "BUY"]
    sell_trades = trades[trades["direction"] == "SELL"]
    buy_pf  = buy_trades[buy_trades["pnl"]>0]["pnl"].sum() / (abs(buy_trades[buy_trades["pnl"]<0]["pnl"].sum())+1e-9)
    sell_pf = sell_trades[sell_trades["pnl"]>0]["pnl"].sum() / (abs(sell_trades[sell_trades["pnl"]<0]["pnl"].sum())+1e-9)

    return {
        "n_trades":        len(trades),
        "n_buy":           len(buy_trades),
        "n_sell":          len(sell_trades),
        "total_pnl":       round(total_pnl, 2),
        "win_rate":        round(win_rate, 4),
        "profit_factor":   round(profit_factor, 3),
        "buy_pf":          round(buy_pf, 3),
        "sell_pf":         round(sell_pf, 3),
        "max_drawdown":    round(max_dd, 2),
        "max_dd_pct":      round(max_dd_pct, 4),
        "sharpe":          round(float(sharpe), 3),
        "expectancy":      round(expectancy, 4),
        "cvar_5pct":       round(cvar_5, 2),
        "avg_win":         round(avg_win, 4),
        "avg_loss":        round(avg_loss, 4),
        "avg_bars_held":   round(trades["bars_held"].mean(), 1),
        "PASS":            _check_pass(profit_factor, win_rate, max_dd_pct, float(sharpe))
    }


def _check_pass(pf, wr, dd_pct, sharpe) -> bool:
    return (
        pf    >= MIN_PROFIT_FACTOR
        and wr    >= MIN_WIN_RATE
        and dd_pct<= MAX_DRAWDOWN_PCT
        and sharpe>= MIN_SHARPE
    )


def print_metrics(metrics: Dict, label: str = ""):
    """Pretty-print metrics table."""
    tag = f"[{label}] " if label else ""
    logger.info(f"\n{'='*60}")
    logger.info(f"{tag}BACKTEST RESULTS")
    logger.info(f"{'='*60}")
    for k, v in metrics.items():
        if k == "error":
            logger.info(f"  ERROR: {v}")
        elif k == "PASS":
            status = "✅ PASS" if v else "❌ FAIL"
            logger.info(f"  Overall: {status}")
        else:
            logger.info(f"  {k:<22}: {v}")
    logger.info(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Purged and embargoed walk-forward cross-validation.
    Prevents data leakage by removing overlap and embargo bars at boundaries.
    """

    def __init__(
        self,
        n_splits:      int = WF_N_SPLITS,
        test_size:     int = WF_TEST_SIZE_BARS,
        embargo_bars:  int = WF_EMBARGO_BARS
    ):
        self.n_splits     = n_splits
        self.test_size    = test_size
        self.embargo_bars = embargo_bars

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Generate (train_idx, test_idx) pairs with embargo between them.
        Train always ends at least embargo_bars before test starts.
        """
        n       = len(df)
        splits  = []
        step    = self.test_size

        for i in range(self.n_splits):
            test_end   = n - i * step
            test_start = test_end - step
            train_end  = test_start - self.embargo_bars

            if train_end < 500:  # Need at least 500 bars to train
                break

            train_idx = df.index[:train_end]
            test_idx  = df.index[test_start:test_end]
            splits.append((train_idx, test_idx))
            logger.info(
                f"WF split {self.n_splits-i}/{self.n_splits}: "
                f"train={train_idx[0].date()}–{train_idx[-1].date()} "
                f"| test={test_idx[0].date()}–{test_idx[-1].date()}"
            )

        return list(reversed(splits))  # Chronological order

    def run(
        self,
        df:           pd.DataFrame,
        X_features:   pd.DataFrame,
        y_buy:        pd.Series,
        y_sell:       pd.Series,
        model_trainer,    # callable: train_fn(X_tr, y_buy_tr, y_sell_tr, X_val, ...) → model
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.60
    ) -> Dict:
        """
        Full walk-forward run. Returns per-split and aggregate metrics.
        """
        splits      = self.split(df)
        all_metrics = []
        all_trades  = []

        for idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"\n{'─'*50}\nWalk-Forward Split {idx+1}/{len(splits)}")

            X_tr   = X_features.loc[train_idx]
            y_b_tr = y_buy.loc[train_idx]
            y_s_tr = y_sell.loc[train_idx]
            X_te   = X_features.loc[test_idx]
            y_b_te = y_buy.loc[test_idx]
            y_s_te = y_sell.loc[test_idx]
            df_te  = df.loc[test_idx]

            # Split train into train/val (last 20% of train = val)
            val_cut  = int(len(X_tr) * 0.8)
            X_val    = X_tr.iloc[val_cut:]
            y_b_val  = y_b_tr.iloc[val_cut:]
            y_s_val  = y_s_tr.iloc[val_cut:]
            X_tr2    = X_tr.iloc[:val_cut]
            y_b_tr2  = y_b_tr.iloc[:val_cut]
            y_s_tr2  = y_s_tr.iloc[:val_cut]

            # Train
            model_stack = model_trainer(X_tr2, y_b_tr2, y_s_tr2, X_val, y_b_val, y_s_val)

            # Predict on test
            buy_probs  = np.array([
                float(model_stack.buy_specialist.predict_proba(X_te.iloc[[i]])[0])
                for i in range(len(X_te))
            ])
            sell_probs = np.array([
                float(model_stack.sell_specialist.predict_proba(X_te.iloc[[i]])[0])
                for i in range(len(X_te))
            ])

            # Simulate
            trades  = simulate_trades(df_te, buy_probs, sell_probs,
                                      buy_threshold, sell_threshold)
            metrics = compute_metrics(trades)
            metrics["split"] = idx + 1
            all_metrics.append(metrics)
            all_trades.append(trades)
            print_metrics(metrics, label=f"Split {idx+1}")

        # Aggregate
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        agg_metrics     = compute_metrics(combined_trades)
        agg_metrics["type"] = "AGGREGATE"
        print_metrics(agg_metrics, label="ALL SPLITS COMBINED")

        return {
            "split_metrics": all_metrics,
            "aggregate":     agg_metrics,
            "all_trades":    combined_trades
        }


# ─────────────────────────────────────────────────────────────────────────────
# STRESS TESTER
# ─────────────────────────────────────────────────────────────────────────────

def run_stress_test(
    df:            pd.DataFrame,
    buy_probs:     np.ndarray,
    sell_probs:    np.ndarray,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.60
) -> Dict:
    """
    Run backtest under stressed conditions:
    - 2× spread
    - Extra slippage
    Compare to baseline to measure cost sensitivity.
    """
    logger.info("Running stress test (2× spread + extra slippage)...")

    baseline = compute_metrics(simulate_trades(
        df, buy_probs, sell_probs, buy_threshold, sell_threshold
    ))
    stressed = compute_metrics(simulate_trades(
        df, buy_probs, sell_probs, buy_threshold, sell_threshold,
        spread_pts  = 20.0 * STRESS_SPREAD_MULT,
        slippage_pts= STRESS_SLIPPAGE_POINTS
    ))

    logger.info(f"Baseline PF={baseline.get('profit_factor'):.3f} | "
                f"Stressed PF={stressed.get('profit_factor'):.3f}")

    return {"baseline": baseline, "stressed": stressed}


# ─────────────────────────────────────────────────────────────────────────────
# REGIME BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def regime_performance_breakdown(
    trades:  pd.DataFrame,
    regimes: pd.Series
) -> pd.DataFrame:
    """
    Show metrics broken down by regime.
    trades:  output of simulate_trades()
    regimes: Series indexed by bar number with regime label
    """
    if trades.empty or regimes.empty:
        return pd.DataFrame()

    rows = []
    for regime_label in regimes.unique():
        mask       = regimes == regime_label
        regime_bars= regimes[mask].index.tolist()
        r_trades   = trades[trades["entry_bar"].isin(regime_bars)]
        if r_trades.empty:
            continue
        m = compute_metrics(r_trades)
        m["regime"] = regime_label
        rows.append(m)

    return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()
