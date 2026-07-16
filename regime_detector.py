"""
=============================================================================
REGIME DETECTOR
5-state HMM regime classification + CUSUM changepoint detection.
Outputs: current regime, regime confidence, transition probability,
         and early warning flag when a regime change is detected.
=============================================================================
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from config import (
    HMM_N_STATES, HMM_N_ITER, HMM_COVARIANCE_TYPE,
    CUSUM_THRESHOLD, CUSUM_DRIFT, REGIME_WINDOW,
    REGIME_LABELS, MODELS_DIR,
    REGIME_USE_TREND_DETECTOR,
    TREND_ADX_PERIOD, TREND_ADX_TREND_THRESH, TREND_ADX_STRONG_THRESH,
    TREND_EMA_FAST, TREND_EMA_MED, TREND_EMA_SLOW, TREND_EMA_TREND,
    TREND_CONF_ADX_WEIGHT, TREND_CONF_ALIGN_WEIGHT,
)

logger = logging.getLogger("RegimeDetector")


# ─────────────────────────────────────────────────────────────────────────────
# REGIME FEATURE BUILDER (inputs to HMM)
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build a compact feature matrix for the HMM regime model.
    Features are designed to distinguish the 5 target regimes.
    Uses only OHLCV-derived features (not the full 50+ feature set).
    """
    out = pd.DataFrame(index=df.index)

    # 1. Return (direction and magnitude)
    ret = df["close"].pct_change()
    out["ret"]       = ret
    out["abs_ret"]   = ret.abs()

    # 2. Realized volatility (5-bar and 20-bar)
    out["rv5"]  = ret.rolling(5).std()
    out["rv20"] = ret.rolling(20).std()
    out["vol_ratio"] = out["rv5"] / (out["rv20"] + 1e-9)  # compression/expansion

    # 3. Trend (EMA divergence)
    ema_fast = df["close"].ewm(span=8, adjust=False).mean()
    ema_slow = df["close"].ewm(span=21, adjust=False).mean()
    out["ema_spread"] = (ema_fast - ema_slow) / (df["close"] + 1e-9)

    # 4. Bar structure (ranging vs trending)
    bar_range = df["high"] - df["low"]
    bar_body  = (df["close"] - df["open"]).abs()
    out["body_ratio"] = bar_body / (bar_range + 1e-9)

    # 5. Rolling autocorrelation (trending markets have positive autocorr)
    out["autocorr"] = ret.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out.dropna(inplace=True)

    return out.values, out.index


# ─────────────────────────────────────────────────────────────────────────────
# HMM REGIME MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HMMRegimeDetector:
    """
    5-state Gaussian HMM for regime detection.
    States map to: TREND_UP, TREND_DOWN, RANGING, VOLATILE, FLAT.
    State ordering is resolved post-fit using mean return per state.
    """

    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.state_map: Dict[int, str] = {}   # HMM state → regime label
        self.fitted  = False

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit HMM on regime features from historical data."""
        logger.info("Fitting HMM regime detector...")
        X_raw, _ = build_regime_features(df)
        X = self.scaler.fit_transform(X_raw)

        self.model = hmm.GaussianHMM(
            n_components    = HMM_N_STATES,
            covariance_type = HMM_COVARIANCE_TYPE,
            n_iter          = HMM_N_ITER,
            random_state    = 42,
            verbose         = False
        )
        self.model.fit(X)

        # Resolve state → label mapping
        self._resolve_state_labels(df, X_raw, X)
        self.fitted = True
        logger.info(f"HMM fitted. State map: {self.state_map}")
        return self

    def _resolve_state_labels(self, df, X_raw, X_scaled):
        """
        Map HMM states to semantic labels based on state characteristics:
        - Mean return → TREND_UP vs TREND_DOWN
        - Volatility   → VOLATILE vs FLAT
        - Body ratio   → RANGING
        """
        states = self.model.predict(X_scaled)
        state_stats = {}
        for s in range(HMM_N_STATES):
            mask = states == s
            mean_ret = X_raw[mask, 0].mean() if mask.sum() > 0 else 0.0
            mean_vol = X_raw[mask, 1].mean() if mask.sum() > 0 else 0.0
            state_stats[s] = {"mean_ret": mean_ret, "mean_vol": mean_vol, "count": mask.sum()}

        sorted_by_ret = sorted(state_stats.keys(), key=lambda s: state_stats[s]["mean_ret"])
        sorted_by_vol = sorted(state_stats.keys(), key=lambda s: state_stats[s]["mean_vol"])

        # Highest mean return → TREND_UP; lowest → TREND_DOWN
        self.state_map[sorted_by_ret[-1]] = "TREND_UP"
        self.state_map[sorted_by_ret[0]]  = "TREND_DOWN"

        # Highest volatility (not already assigned) → VOLATILE
        for s in reversed(sorted_by_vol):
            if s not in self.state_map:
                self.state_map[s] = "VOLATILE"
                break

        # Lowest volatility (not assigned) → FLAT
        for s in sorted_by_vol:
            if s not in self.state_map:
                self.state_map[s] = "FLAT"
                break

        # Remaining → RANGING
        for s in range(HMM_N_STATES):
            if s not in self.state_map:
                self.state_map[s] = "RANGING"

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime for each bar in df.
        Returns DataFrame with: regime_state, regime_label, regime_conf, transition_prob.
        """
        if not self.fitted:
            raise RuntimeError("HMM not fitted. Call fit() first.")

        X_raw, index = build_regime_features(df)
        if len(X_raw) == 0:
            return pd.DataFrame()

        X = self.scaler.transform(X_raw)
        states = self.model.predict(X)
        log_probs = self.model.predict_proba(X)

        labels = [self.state_map.get(s, "UNKNOWN") for s in states]
        confs  = log_probs.max(axis=1)

        # Transition probability (probability of moving to a different state next bar)
        trans_mat = self.model.transmat_
        trans_probs = np.array([1 - trans_mat[s, s] for s in states])

        result = pd.DataFrame({
            "regime_state":   states,
            "regime_label":   labels,
            "regime_conf":    confs,
            "regime_trans_p": trans_probs,
        }, index=index)

        return result

    def get_current_regime(self, df: pd.DataFrame) -> dict:
        """Get regime for the most recent bar."""
        pred = self.predict(df.tail(REGIME_WINDOW + 10))
        if pred.empty:
            return {"regime": "UNKNOWN", "confidence": 0.0, "transition_risk": 1.0}
        last = pred.iloc[-1]
        return {
            "regime":          last["regime_label"],
            "confidence":      float(last["regime_conf"]),
            "transition_risk": float(last["regime_trans_p"]),
        }

    def save(self, path: Optional[str] = None):
        path = path or os.path.join(MODELS_DIR, "hmm_regime.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "state_map": self.state_map}, f)
        logger.info(f"HMM saved: {path}")

    def load(self, path: Optional[str] = None) -> "HMMRegimeDetector":
        path = path or os.path.join(MODELS_DIR, "hmm_regime.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model     = data["model"]
        self.scaler    = data["scaler"]
        self.state_map = data["state_map"]
        self.fitted    = True
        logger.info(f"HMM loaded: {path}. State map: {self.state_map}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# CUSUM CHANGEPOINT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) control chart for early regime transition detection.
    Detects when the mean of a process has shifted significantly.
    Raises a changepoint flag BEFORE the HMM state changes — providing
    early warning to tighten thresholds or pause trading.
    """

    def __init__(
        self,
        threshold: float = CUSUM_THRESHOLD,
        drift:     float = CUSUM_DRIFT
    ):
        self.threshold = threshold
        self.drift     = drift
        self._reset()

    def _reset(self):
        self.pos_cusum  = 0.0
        self.neg_cusum  = 0.0
        self.target_mean = None
        self.target_std  = None

    def calibrate(self, series: pd.Series, warmup: int = 50):
        """Set the reference mean and std from the first `warmup` observations."""
        self.target_mean = series.iloc[:warmup].mean()
        self.target_std  = series.iloc[:warmup].std() + 1e-9

    def update(self, value: float) -> bool:
        """
        Update CUSUM with new observation. Returns True if changepoint detected.
        """
        if self.target_mean is None:
            return False
        z = (value - self.target_mean) / self.target_std
        self.pos_cusum = max(0, self.pos_cusum + z - self.drift)
        self.neg_cusum = max(0, self.neg_cusum - z - self.drift)
        return self.pos_cusum > self.threshold or self.neg_cusum > self.threshold

    def reset_after_detection(self):
        """Reset accumulators after changepoint signal is acted upon."""
        self.pos_cusum = 0.0
        self.neg_cusum = 0.0

    def compute_series(self, series: pd.Series, warmup: int = 50) -> pd.DataFrame:
        """
        Compute CUSUM statistics for entire series.
        Returns DataFrame with: cusum_pos, cusum_neg, changepoint flag.
        """
        self.calibrate(series, warmup)
        pos_vals    = []
        neg_vals    = []
        cp_flags    = []
        pos_cusum   = 0.0
        neg_cusum   = 0.0
        m           = series.mean()
        s           = series.std() + 1e-9

        for v in series.values:
            z = (v - m) / s
            pos_cusum = max(0, pos_cusum + z - self.drift)
            neg_cusum = max(0, neg_cusum - z - self.drift)
            detected  = pos_cusum > self.threshold or neg_cusum > self.threshold
            pos_vals.append(pos_cusum)
            neg_vals.append(neg_cusum)
            cp_flags.append(int(detected))
            if detected:
                pos_cusum = 0.0
                neg_cusum = 0.0
                m = v
                s = series.std() + 1e-9

        return pd.DataFrame({
            "cusum_pos":    pos_vals,
            "cusum_neg":    neg_vals,
            "changepoint":  cp_flags
        }, index=series.index)


# ─────────────────────────────────────────────────────────────────────────────
# TREND-BASED REGIME DETECTOR (ADX + multi-EMA, no training required)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(df_col: pd.Series, span: int) -> pd.Series:
    return df_col.ewm(span=span, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute ADX, +DI, -DI using Wilder's smoothing.
    Returns (adx, plus_di, minus_di) series aligned to input index.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1).fillna(0.0)

    # Wilder's smoothing (EMA with alpha=1/period)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100.0 * plus_di_smooth / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_di_smooth / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


class TrendRegimeDetector:
    """
    Fast-adapting trend-based regime detector using ADX and multi-EMA analysis.

    Replaces the HMM with a deterministic, zero-training approach:
    - ADX measures trend STRENGTH
    - Multi-period EMA alignment measures trend DIRECTION across timeframes
    - Volatility (ATR percentile) distinguishes VOLATILE from RANGING

    Uses the same get_regime() interface as RegimeRouter so it is a
    drop-in replacement for live_trader.py and replay scripts.
    """

    def __init__(self):
        self._prev_atr_ret = None

    # ── Public interface (matches RegimeRouter) ─────────────────────────────

    def fit(self, df: pd.DataFrame) -> "TrendRegimeDetector":
        return self   # no training needed

    def load(self, path: Optional[str] = None) -> "TrendRegimeDetector":
        return self   # no persisted model

    def save(self, path: Optional[str] = None):
        pass          # no persisted model

    def get_regime(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < max(TREND_ADX_PERIOD + 5, TREND_EMA_TREND + 5):
            return {
                "regime": "UNKNOWN", "confidence": 0.0,
                "transition_risk": 1.0, "cusum_warning": False, "trade_ok": False,
            }

        adx, plus_di, minus_di = _adx(df["high"], df["low"], df["close"], TREND_ADX_PERIOD)
        adx_val = float(adx.iloc[-1])
        di_plus = float(plus_di.iloc[-1])
        di_minus = float(minus_di.iloc[-1])
        di_bull = di_plus > di_minus  # +DI above -DI = bullish

        # Multi-EMA directions (1 = up, -1 = down, 0 = flat)
        close = df["close"].astype(float)
        ema_fast = float(_ema(close, TREND_EMA_FAST).iloc[-1])
        ema_med = float(_ema(close, TREND_EMA_MED).iloc[-1])
        ema_slow = float(_ema(close, TREND_EMA_SLOW).iloc[-1])
        ema_trend = float(_ema(close, TREND_EMA_TREND).iloc[-1])
        price = float(close.iloc[-1])

        def _direction(val, ref) -> int:
            return 1 if val > ref * 1.001 else -1 if val < ref * 0.999 else 0

        dir_fast = _direction(price, ema_fast)
        dir_med = _direction(price, ema_med)
        dir_slow = _direction(price, ema_slow)
        dir_trend = _direction(price, ema_trend)
        dir_ema_cross = _direction(ema_fast, ema_med)

        # Compute medium-term price slope (21-bar linear approx)
        slope_bars = min(21, len(close) - 1)
        slope = float(close.iloc[-1] - close.iloc[-1 - slope_bars]) / slope_bars
        dir_slope = 1 if slope > 0 else -1 if slope < 0 else 0

        # Direction votes weighted by timeframe significance
        votes = [
            dir_fast * 1,        # short-term
            dir_med * 2,        # medium-term
            dir_slow * 3,       # long-term
            dir_trend * 3,      # major trend
            dir_ema_cross * 2,  # EMA cross signal
            dir_slope * 1,      # price slope
            (1 if di_bull else -1) * (1 if adx_val > 20 else 0),  # DI direction (only when ADX > 20)
        ]
        vote_sum = sum(votes)
        vote_abs = sum(abs(v) for v in votes)
        alignment = vote_sum / vote_abs if vote_abs > 0 else 0.0  # -1 to 1

        # Trend strength from ADX normalized to [0, 1]
        adx_norm = min(1.0, adx_val / 60.0)

        # Confidence: blend ADX strength + multi-TF alignment
        confidence = (
            TREND_CONF_ADX_WEIGHT * adx_norm
            + TREND_CONF_ALIGN_WEIGHT * abs(alignment)
        )
        confidence = min(1.0, max(0.0, confidence))

        # Volatility regime check
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        atr_percentile = 0.5
        if len(df) > 100:
            atr_hist = (df["high"] - df["low"]).rolling(14).mean().dropna()
            atr_percentile = (atr_hist.rank(pct=True).iloc[-1]) if len(atr_hist) > 0 else 0.5
        high_vol = float(atr_percentile) > 0.80

        # Determine regime label
        is_trending = adx_val >= TREND_ADX_TREND_THRESH
        is_strong_trend = adx_val >= TREND_ADX_STRONG_THRESH

        if is_trending:
            if alignment > 0.3:
                regime = "TREND_UP"
                confidence = min(1.0, confidence * (1.2 if is_strong_trend else 1.0))
            elif alignment < -0.3:
                regime = "TREND_DOWN"
                confidence = min(1.0, confidence * (1.2 if is_strong_trend else 1.0))
            elif alignment > 0:
                regime = "TREND_UP"
            else:
                regime = "TREND_DOWN"
        elif high_vol:
            regime = "VOLATILE"
            confidence *= 0.7
        else:
            regime = "RANGING"
            confidence *= 0.4

        # Transition risk: how fast is ADX changing
        if len(adx) > 3:
            adx_delta = float(adx.iloc[-1] - adx.iloc[-4]) / (TREND_ADX_PERIOD)
            transition_risk = min(1.0, max(0.0, abs(adx_delta)))
        else:
            transition_risk = 0.5

        trade_ok = True   # caller gates on REGIME_MIN_CONFIDENCE; no redundant hard threshold here

        return {
            "regime":          regime,
            "confidence":      round(confidence, 4),
            "transition_risk": round(transition_risk, 4),
            "cusum_warning":   False,
            "trade_ok":        trade_ok,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Option 2: MARKET-STRUCTURE BREAKOUT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_market_breakout(
    m1_df: Optional[pd.DataFrame] = None,
    lookback_bars: int = 60,
    vol_mult: float = 1.3,
    min_strength: float = 0.5,
) -> Optional[Dict]:
    """
    Detect if price has broken the 1-hour range with volume confirmation.
    Returns a dict with 'direction', 'strength', 'confidence' on success,
    or None if no breakout is detected.

    Parameters
    ----------
    m1_df : optional pre-fetched M1 DataFrame. If None, tries MT5.
    lookback_bars : number of M1 bars for the range window (default 60 = 1h).
    vol_mult : volume multiple above rolling average required.
    min_strength : minimum normalized breakout strength to qualify.
    """
    if m1_df is None:
        try:
            import MetaTrader5 as mt5
            from datetime import datetime, timezone, timedelta
            from config import SYMBOL
            now = datetime.now(timezone.utc)
            rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1,
                                         now - timedelta(minutes=lookback_bars + 10),
                                         now)
            if rates is None or len(rates) == 0:
                return None
            m1_df = pd.DataFrame(rates)
            m1_df["time"] = pd.to_datetime(m1_df["time"], unit="s")
            m1_df.set_index("time", inplace=True)
        except Exception:
            return None

    if m1_df is None or len(m1_df) < lookback_bars // 2:
        return None

    # Use closed bars only (exclude the forming bar if present)
    df = m1_df.copy()
    df = df.iloc[:-(1 if len(df) > lookback_bars else 0)]

    if len(df) < lookback_bars // 2:
        return None

    # Range: high/low of everything except the last few bars
    range_bars = df.iloc[:-(max(3, lookback_bars // 20))]
    if range_bars.empty:
        return None
    range_high = float(range_bars["high"].max())
    range_low  = float(range_bars["low"].min())
    range_size = max(range_high - range_low, 0.01)

    # Current close (last closed bar)
    current = df.iloc[-1]
    price   = float(current["close"])

    # Volume check
    vol_series = df["tick_volume"].astype(float)
    avg_vol    = float(vol_series.iloc[-(lookback_bars // 2):-1].mean())
    cur_vol    = float(vol_series.iloc[-1])
    vol_ok     = cur_vol > avg_vol * vol_mult if avg_vol > 0 else True

    if not vol_ok:
        # Volume check failed on closed bars — try forming bar instead
        if len(m1_df) > len(df):
            forming = m1_df.iloc[-1]
            forming_high = float(forming["high"])
            forming_low  = float(forming["low"])
            forming_vol  = float(forming["tick_volume"])
            # Use forming bar's extreme + partial volume as proxy
            if forming_vol > avg_vol * vol_mult * 0.7:
                if forming_high > range_high:
                    strength = min(1.0, (forming_high - range_high) / (range_size * 0.5))
                    if strength >= min_strength:
                        return {
                            "direction": "BUY",
                            "strength":  round(strength, 3),
                            "confidence": round(0.55 + strength * 0.35, 3),
                            "price":     float(forming["close"]),
                            "range_high": range_high,
                            "range_low":  range_low,
                        }
                elif forming_low < range_low:
                    strength = min(1.0, (range_low - forming_low) / (range_size * 0.5))
                    if strength >= min_strength:
                        return {
                            "direction": "SELL",
                            "strength":  round(strength, 3),
                            "confidence": round(0.55 + strength * 0.35, 3),
                            "price":     float(forming["close"]),
                            "range_high": range_high,
                            "range_low":  range_low,
                        }
        return None

    # Breakout direction and strength
    if price > range_high:
        strength = min(1.0, (price - range_high) / (range_size * 0.5))
        if strength >= min_strength:
            return {
                "direction": "BUY",
                "strength":  round(strength, 3),
                "confidence": round(0.50 + strength * 0.40, 3),
                "price":     price,
                "range_high": range_high,
                "range_low":  range_low,
            }
    elif price < range_low:
        strength = min(1.0, (range_low - price) / (range_size * 0.5))
        if strength >= min_strength:
            return {
                "direction": "SELL",
                "strength":  round(strength, 3),
                "confidence": round(0.50 + strength * 0.40, 3),
                "price":     price,
                "range_high": range_high,
                "range_low":  range_low,
            }

    # Also check forming bar for extreme wick breakout (catch spike mid-bar)
    if len(m1_df) > len(df):
        forming = m1_df.iloc[-1]
        forming_high = float(forming["high"])
        forming_low  = float(forming["low"])
        forming_vol  = float(forming["tick_volume"])
        vol_spike = forming_vol > avg_vol * vol_mult * 0.7 if avg_vol > 0 else True
        if vol_spike and forming_high > range_high and not price > range_high:
            strength = min(1.0, (forming_high - range_high) / (range_size * 0.5))
            if strength >= min_strength:
                return {
                    "direction": "BUY",
                    "strength":  round(strength, 3),
                    "confidence": round(0.50 + strength * 0.35, 3),
                    "price":     float(forming["close"]),
                    "range_high": range_high,
                    "range_low":  range_low,
                }
        if vol_spike and forming_low < range_low and not price < range_low:
            strength = min(1.0, (range_low - forming_low) / (range_size * 0.5))
            if strength >= min_strength:
                return {
                    "direction": "SELL",
                    "strength":  round(strength, 3),
                    "confidence": round(0.50 + strength * 0.35, 3),
                    "price":     float(forming["close"]),
                    "range_high": range_high,
                    "range_low":  range_low,
                }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Option 3: TREND STRUCTURE CHECK (ADX + multi-EMA alignment)
# ─────────────────────────────────────────────────────────────────────────────

def check_trend_structure(
    df_m15: pd.DataFrame,
    adx_min: int = 25,
    max_atr_extension: float = 0.0,
) -> Optional[str]:
    """
    Check if M15 ADX and multi-EMA alignment indicate a clear trend.
    Returns "BUY", "SELL", or None if no clear trend.

    Parameters
    ----------
    df_m15 : M15 OHLC DataFrame with enough history for warmup.
    adx_min : minimum ADX value to consider trending.
    max_atr_extension : if > 0, block when price is more than this many
                        ATRs from the 200 EMA (prevents buying after a
                        massive spike / entering at price extremes).
    """
    if df_m15 is None or len(df_m15) < max(TREND_ADX_PERIOD + 5, TREND_EMA_TREND + 5):
        return None

    adx, plus_di, minus_di = _adx(
        df_m15["high"], df_m15["low"], df_m15["close"], TREND_ADX_PERIOD
    )
    adx_val = float(adx.iloc[-1])
    if adx_val < adx_min:
        return None

    close = df_m15["close"].astype(float)
    ema_fast = float(_ema(close, TREND_EMA_FAST).iloc[-1])
    ema_med  = float(_ema(close, TREND_EMA_MED).iloc[-1])
    ema_slow = float(_ema(close, TREND_EMA_SLOW).iloc[-1])
    price    = float(close.iloc[-1])
    di_plus  = float(plus_di.iloc[-1])
    di_minus = float(minus_di.iloc[-1])

    # Price-extension check: if max_atr_extension > 0, compute distance
    # from 200 EMA and block if price is too extended.
    if max_atr_extension > 0 and len(df_m15) >= TREND_EMA_TREND + 5:
        ema_trend = float(_ema(close, TREND_EMA_TREND).iloc[-1])
        hl = df_m15["high"] - df_m15["low"]
        hc = (df_m15["high"] - df_m15["close"].shift(1)).abs()
        lc = (df_m15["low"]  - df_m15["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1).fillna(0)
        atr_val = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
        if atr_val > 0:
            dist = abs(price - ema_trend) / atr_val
            if dist > max_atr_extension:
                return None

    # Bullish: price above all EMAs, EMAs stacked, +DI > -DI
    if price > ema_slow and ema_fast > ema_med > ema_slow and di_plus > di_minus:
        return "BUY"

    # Bearish: price below all EMAs, EMAs stacked, -DI > +DI
    if price < ema_slow and ema_fast < ema_med < ema_slow and di_minus > di_plus:
        return "SELL"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Option 4: PRICE-ACTION DIRECTION FILTER
# ─────────────────────────────────────────────────────────────────────────────

def price_action_direction(
    df_m15: pd.DataFrame,
    min_swing_bars: int = 5,
    ema_weight: float = 0.40,
    di_weight: float = 0.30,
    swing_weight: float = 0.30,
) -> Dict:
    """
    Standalone price-action direction filter.

    Combines three independent signals into a composite direction score:
      1. Multi-EMA alignment (8/21/55/200)        — weight ema_weight
      2. DI+/DI- crossover from ADX                — weight di_weight
      3. Swing-point structure (higher highs/lows) — weight swing_weight

    Returns
    -------
    dict with keys:
        direction : "BUY" | "SELL" | "NEUTRAL"
        confidence: float 0..1 (how strongly the filter believes in the direction)
        score     : float -1 (strong SELL) .. +1 (strong BUY)
        components: dict of individual sub-scores
    """
    result = {"direction": "NEUTRAL", "confidence": 0.0, "score": 0.0,
              "components": {}}
    if df_m15 is None or len(df_m15) < max(60, TREND_EMA_TREND + 5):
        return result

    close = df_m15["close"].astype(float)
    high  = df_m15["high"].astype(float)
    low   = df_m15["low"].astype(float)
    price = float(close.iloc[-1])

    # ── 1. Multi-EMA alignment score ──────────────────────────────────────
    ema8   = float(_ema(close, TREND_EMA_FAST).iloc[-1])
    ema21  = float(_ema(close, TREND_EMA_MED).iloc[-1])
    ema55  = float(_ema(close, TREND_EMA_SLOW).iloc[-1])
    ema200 = float(_ema(close, TREND_EMA_TREND).iloc[-1]) \
             if len(df_m15) >= TREND_EMA_TREND + 5 else None

    ema_score = 0.0
    bull_stacked = ema8 > ema21 > ema55
    bear_stacked = ema8 < ema21 < ema55
    if ema200 is not None:
        bull_stacked = bull_stacked and price > ema200
        bear_stacked = bear_stacked and price < ema200

    if bull_stacked and price > ema55:
        ema_score = 1.0
    elif price > ema55 and ema8 > ema21:
        ema_score = 0.5
    elif bear_stacked and price < ema55:
        ema_score = -1.0
    elif price < ema55 and ema8 < ema21:
        ema_score = -0.5

    # ── 2. DI+/DI- crossover score ───────────────────────────────────────
    adx_series, plus_di, minus_di = _adx(high, low, close, TREND_ADX_PERIOD)
    di_plus  = float(plus_di.iloc[-1])
    di_minus = float(minus_di.iloc[-1])
    di_margin = di_plus - di_minus

    di_score = 0.0
    if di_margin > 5.0:
        di_score = 1.0
    elif di_margin > 0:
        di_score = 0.5
    elif di_margin < -5.0:
        di_score = -1.0
    elif di_margin < 0:
        di_score = -0.5

    # ── 3. Swing-point structure score ────────────────────────────────────
    # Compare the last two windows of `min_swing_bars` bars:
    #   HH = most-recent window high > prior window high
    #   HL = most-recent window low  > prior window low
    #   LH = most-recent window high < prior window high
    #   LL = most-recent window low  < prior window low
    swing_score = 0.0
    if len(df_m15) >= min_swing_bars * 3:
        w2_high = float(high.iloc[-min_swing_bars*2:-min_swing_bars].max())
        w2_low  = float(low.iloc[-min_swing_bars*2:-min_swing_bars].min())
        w1_high = float(high.iloc[-min_swing_bars:].max())
        w1_low  = float(low.iloc[-min_swing_bars:].min())

        hh = w1_high > w2_high
        hl = w1_low  > w2_low
        lh = w1_high < w2_high
        ll = w1_low  < w2_low

        if hh and hl:
            swing_score = 1.0
        elif hh or hl:
            swing_score = 0.5
        elif lh and ll:
            swing_score = -1.0
        elif lh or ll:
            swing_score = -0.5

    # ── Composite score ───────────────────────────────────────────────────
    total_weight = ema_weight + di_weight + swing_weight
    if total_weight <= 0:
        return result
    composite = (ema_weight * ema_score + di_weight * di_score
                 + swing_weight * swing_score) / total_weight

    result["components"] = {
        "ema_score":   round(ema_score,   2),
        "di_score":    round(di_score,    2),
        "swing_score": round(swing_score, 2),
        "composite":   round(composite,   2),
    }

    if composite >= 0.50:
        result["direction"] = "BUY"
        result["confidence"] = round(min(composite, 1.0), 2)
    elif composite <= -0.50:
        result["direction"] = "SELL"
        result["confidence"] = round(min(abs(composite), 1.0), 2)
    else:
        result["direction"] = "NEUTRAL"
        result["confidence"] = round(abs(composite), 2)
    result["score"] = round(composite, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# M30 DIRECTION FILTER — Independent higher-trend direction with momentum/ROC
# ─────────────────────────────────────────────────────────────────────────────

def m30_direction_filter(
    df_m30: pd.DataFrame,
    min_swing_bars: int = 5,
    ema_weight: float = 0.25,
    momentum_weight: float = 0.30,
    accel_weight: float = 0.20,
    di_weight: float = 0.15,
    swing_weight: float = 0.10,
) -> Dict:
    """
    Independent direction filter designed for M30 (30-min) data.

    Combines five signals into a composite direction score:
      1. Multi-EMA alignment (8/21/55)    — weight ema_weight
      2. ROC momentum (3/5/10-bar)         — weight momentum_weight
      3. ROC acceleration/deceleration     — weight accel_weight
      4. DI+/DI- crossover                 — weight di_weight
      5. Swing-point structure             — weight swing_weight

    M30 is chosen over H1 for GOLD because intraday trends run 2-6 hours;
    M30 confirms in 4-6 bars vs H1's 2-3, catching the same trend earlier.
    M30 ATR (~$15-20) also keeps flip_wait_price reachable within 90-min
    queue expiry, whereas H1 ATR (~$25-30) is often too far.

    Returns
    -------
    dict with keys:
        direction : "BUY" | "SELL" | "NEUTRAL"
        confidence: float 0..1
        score     : float -1 (strong SELL) .. +1 (strong BUY)
        components: dict of individual sub-scores
    """
    result = {"direction": "NEUTRAL", "confidence": 0.0, "score": 0.0,
              "components": {}}
    if df_m30 is None or len(df_m30) < 40:
        return result

    close = df_m30["close"].astype(float)
    high  = df_m30["high"].astype(float)
    low   = df_m30["low"].astype(float)

    # ── 1. Multi-EMA alignment score ──────────────────────────────────────
    ema8   = float(_ema(close, 8).iloc[-1])
    ema21  = float(_ema(close, 21).iloc[-1])
    ema55  = float(_ema(close, 55).iloc[-1])
    px     = float(close.iloc[-1])

    ema_score = 0.0
    if ema8 > ema21 > ema55 and px > ema55:
        ema_score = 1.0
    elif px > ema55 and ema8 > ema21:
        ema_score = 0.7
    elif px > ema21 and ema8 > ema21:
        ema_score = 0.4
    elif px > ema8 and ema8 > ema21:
        ema_score = 0.2
    elif ema8 < ema21 < ema55 and px < ema55:
        ema_score = -1.0
    elif px < ema55 and ema8 < ema21:
        ema_score = -0.7
    elif px < ema21 and ema8 < ema21:
        ema_score = -0.4
    elif px < ema8 and ema8 < ema21:
        ema_score = -0.2

    # ── 2. Momentum score (tanh-scaled ROC) ───────────────────────────────
    momentum_score = 0.0
    if len(close) >= 15:
        roc3   = close.pct_change(3) * 100.0
        roc5   = close.pct_change(5) * 100.0
        roc10  = close.pct_change(10) * 100.0
        roc_avg = float(((roc3 + roc5 + roc10) / 3.0).iloc[-1])
        momentum_score = float(np.tanh(roc_avg * 5.0))

    # ── 3. Acceleration score (tanh-scaled per-bar rate change) ───────────
    accel_score = 0.0
    if len(close) >= 15:
        roc3   = float(close.pct_change(3).iloc[-1] * 100.0 / 3.0)
        roc10  = float(close.pct_change(10).iloc[-1] * 100.0 / 10.0)
        accel_raw = roc3 - roc10
        accel_score = float(np.tanh(accel_raw * 10.0))

    # ── 4. DI+/DI- crossover score ───────────────────────────────────────
    adx_series, plus_di, minus_di = _adx(high, low, close, 14)
    di_plus  = float(plus_di.iloc[-1])
    di_minus = float(minus_di.iloc[-1])
    di_margin = di_plus - di_minus

    di_score = 0.0
    if di_margin > 3.0:
        di_score = 1.0
    elif di_margin > 0:
        di_score = 0.5
    elif di_margin < -3.0:
        di_score = -1.0
    elif di_margin < 0:
        di_score = -0.5

    # ── 5. Swing-point structure score ────────────────────────────────────
    swing_score = 0.0
    if len(df_m30) >= min_swing_bars * 3:
        w2_high = float(high.iloc[-min_swing_bars*2:-min_swing_bars].max())
        w2_low  = float(low.iloc[-min_swing_bars*2:-min_swing_bars].min())
        w1_high = float(high.iloc[-min_swing_bars:].max())
        w1_low  = float(low.iloc[-min_swing_bars:].min())

        hh = w1_high > w2_high
        hl = w1_low  > w2_low
        lh = w1_high < w2_high
        ll = w1_low  < w2_low

        if hh and hl:
            swing_score = 1.0
        elif hh or hl:
            swing_score = 0.5
        elif lh and ll:
            swing_score = -1.0
        elif lh or ll:
            swing_score = -0.5

    # ── Composite score with divergence dampening ────────────────────────
    # "Slow" components (EMA, DI, swing) capture lagging structure.
    # "Fast" components (momentum, accel) capture live price velocity.
    # When fast components disagree with or are much weaker than slow ones,
    # the trend is mature/stalling — halve confidence to prevent false flips.
    total_weight = ema_weight + momentum_weight + accel_weight + di_weight + swing_weight
    if total_weight <= 0:
        return result
    composite = (ema_weight * ema_score + momentum_weight * momentum_score
                 + accel_weight * accel_score
                 + di_weight * di_score + swing_weight * swing_score) / total_weight

    slow_w = ema_weight + di_weight + swing_weight
    fast_w = momentum_weight + accel_weight
    if slow_w > 0 and fast_w > 0:
        slow_dir = (ema_weight * ema_score + di_weight * di_score
                    + swing_weight * swing_score) / slow_w
        fast_dir = (momentum_weight * momentum_score
                    + accel_weight * accel_score) / fast_w
        if abs(fast_dir) < 0.10:
            composite *= 0.7
        elif slow_dir * fast_dir < 0:
            composite *= 0.5

    result["components"] = {
        "ema_score":       round(ema_score, 2),
        "momentum_score":  round(momentum_score, 2),
        "accel_score":     round(accel_score, 2),
        "di_score":        round(di_score, 2),
        "swing_score":     round(swing_score, 2),
        "composite":       round(composite, 2),
    }

    if composite >= 0.40:
        result["direction"] = "BUY"
        result["confidence"] = round(min(composite, 1.0), 2)
    elif composite <= -0.40:
        result["direction"] = "SELL"
        result["confidence"] = round(min(abs(composite), 1.0), 2)
    else:
        result["direction"] = "NEUTRAL"
        result["confidence"] = round(abs(composite), 2)
    result["score"] = round(composite, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED REGIME ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class RegimeRouter:
    """
    Combines HMM regime classification with CUSUM changepoint detection.
    Provides a single interface for the meta-agent to query:
        - Current regime
        - Regime confidence
        - Transition risk (CUSUM warning active?)
        - Whether to use tighter thresholds
    """

    def __init__(self):
        self.hmm    = HMMRegimeDetector()
        self.cusum  = CUSUMDetector()
        self._cusum_calibrated = False

    def fit(self, df: pd.DataFrame) -> "RegimeRouter":
        """Fit HMM and calibrate CUSUM on training data."""
        self.hmm.fit(df)

        # Use ATR-normalized returns as CUSUM target series
        atr = (df["high"] - df["low"]).ewm(span=14, adjust=False).mean()
        atr_ret = df["close"].pct_change() / (atr / df["close"] + 1e-9)
        self.cusum.calibrate(atr_ret.dropna(), warmup=100)
        self._cusum_calibrated = True
        return self

    def get_regime(self, df: pd.DataFrame) -> dict:
        """
        Full regime assessment for the current bar.

        Returns:
            regime       : str ("TREND_UP", "TREND_DOWN", "RANGING", "VOLATILE", "FLAT", "UNKNOWN")
            confidence   : float [0, 1]
            cusum_warning: bool (True if changepoint detected — tighten thresholds)
            trade_ok     : bool (False if confidence too low or CUSUM in alert)
        """
        # HMM regime
        hmm_result = self.hmm.get_current_regime(df)
        regime     = hmm_result["regime"]
        confidence = hmm_result["confidence"]
        trans_risk = hmm_result["transition_risk"]

        # CUSUM changepoint check
        atr = (df["high"] - df["low"]).ewm(span=14, adjust=False).mean()
        atr_ret = df["close"].pct_change() / (atr / df["close"] + 1e-9)
        latest_atr_ret = atr_ret.iloc[-1] if not atr_ret.empty else 0.0
        cusum_alert = self.cusum.update(float(latest_atr_ret))

        if cusum_alert:
            self.cusum.reset_after_detection()
            logger.warning(f"CUSUM changepoint detected. Tightening thresholds.")

        # Trade allowed if regime is confident and no transition in progress
        trade_ok = (
            regime != "UNKNOWN"
            and confidence >= 0.50
            and not (cusum_alert and confidence < 0.65)
        )

        return {
            "regime":        regime,
            "confidence":    confidence,
            "transition_risk": trans_risk,
            "cusum_warning": cusum_alert,
            "trade_ok":      trade_ok,
        }

    def save(self, directory: str = MODELS_DIR):
        self.hmm.save(os.path.join(directory, "hmm_regime.pkl"))
        cusum_path = os.path.join(directory, "cusum_state.pkl")
        with open(cusum_path, "wb") as f:
            pickle.dump(self.cusum, f)
        logger.info("RegimeRouter saved.")

    def load(self, directory: str = MODELS_DIR) -> "RegimeRouter":
        self.hmm.load(os.path.join(directory, "hmm_regime.pkl"))
        cusum_path = os.path.join(directory, "cusum_state.pkl")
        if os.path.exists(cusum_path):
            with open(cusum_path, "rb") as f:
                self.cusum = pickle.load(f)
            self._cusum_calibrated = True
        logger.info("RegimeRouter loaded.")
        return self
