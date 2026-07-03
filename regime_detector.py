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
    REGIME_LABELS, MODELS_DIR
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
