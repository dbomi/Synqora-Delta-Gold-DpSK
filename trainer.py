"""
=============================================================================
TRAINER — SYNQORA DELTA GOLD FABLE
End-to-end training pipeline for the Fable model set:

  1. Fetch multi-TF historical data (MT5)
  2. Build features
  3. Build triple-barrier labels (TP=2.0×ATR, SL=1.0×ATR, 48 bars)
  4. Fit HMM + CUSUM regime router
  5. Train GBM BUY / SELL specialists
  6. Walk-forward validation
  7. Stress test

No LSTM, no meta-learner — only the validated model set is trained.
To retrain ONLY the regime intelligence (keeping the shipped GBM models),
run retrain_hmm.py instead.
=============================================================================
"""

import logging
import os
import pickle
from datetime import datetime

from config import (
    SETUP_TAG, SYMBOL, PRIMARY_TF, CONTEXT_TFS,
    TRAIN_START, TRAIN_END,
    MODELS_DIR, LOGS_DIR,
    TRIPLE_BARRIER_TP_ATR, TRIPLE_BARRIER_SL_ATR, TRIPLE_BARRIER_MAX_BARS,
)
from data_engine import initialize_mt5, shutdown_mt5, fetch_multi_tf, align_to_primary
from feature_engine import build_features, get_feature_columns
from label_engine import build_labels
from regime_detector import RegimeRouter
from model_stack import GBMSpecialist, ModelStack
from validation_engine import WalkForwardValidator, run_stress_test

# ── Logging setup ──────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    handlers= [
        logging.FileHandler(os.path.join(LOGS_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.log"),
                            mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Trainer")


def step_fetch_data():
    logger.info("="*60)
    logger.info("STEP 1: Fetching historical data")
    logger.info("="*60)

    start = datetime.strptime(TRAIN_START, "%Y-%m-%d")
    end   = datetime.strptime(TRAIN_END,   "%Y-%m-%d")

    data = fetch_multi_tf(SYMBOL, PRIMARY_TF, CONTEXT_TFS, start, end, use_cache=True)
    if PRIMARY_TF not in data or data[PRIMARY_TF].empty:
        raise RuntimeError(f"Failed to fetch primary TF data ({PRIMARY_TF})")

    data_aligned = align_to_primary(data, PRIMARY_TF)
    logger.info(f"Data fetched: {len(data_aligned[PRIMARY_TF])} bars on {PRIMARY_TF}")
    return data_aligned


def step_build_features(data_aligned):
    logger.info("="*60)
    logger.info("STEP 2: Building features")
    logger.info("="*60)

    df_features = build_features(data_aligned, PRIMARY_TF)
    feat_cols   = get_feature_columns(df_features)
    logger.info(f"Features built: {len(feat_cols)} columns | {len(df_features)} bars")
    return df_features, feat_cols


def step_build_labels(df_features, data_aligned):
    logger.info("="*60)
    logger.info("STEP 3: Building labels")
    logger.info("="*60)

    ohlcv_cols = ["open", "high", "low", "close", "tick_volume"]
    df_primary = data_aligned[PRIMARY_TF]
    df_for_labels = df_features.copy()
    for col in ohlcv_cols:
        if col in df_primary.columns and col not in df_for_labels.columns:
            df_for_labels[col] = df_primary[col].reindex(df_for_labels.index)

    df_labeled = build_labels(
        df_for_labels,
        tp_atr_mult = TRIPLE_BARRIER_TP_ATR,
        sl_atr_mult = TRIPLE_BARRIER_SL_ATR,
        max_bars    = TRIPLE_BARRIER_MAX_BARS,
        compute_mfe = True
    )
    logger.info(f"Labels built: {len(df_labeled)} bars")
    return df_labeled


def step_fit_regime(data_aligned):
    logger.info("="*60)
    logger.info("STEP 4: Fitting HMM regime router")
    logger.info("="*60)

    router = RegimeRouter()
    router.fit(data_aligned[PRIMARY_TF])
    router.save(MODELS_DIR)
    logger.info("Regime router fitted and saved.")
    return router


def step_train_specialists(df_labeled, feat_cols):
    logger.info("="*60)
    logger.info("STEP 5: Training GBM specialists")
    logger.info("="*60)

    X = df_labeled[feat_cols].copy()
    y_buy  = df_labeled["buy_win"]
    y_sell = df_labeled["sell_win"]

    split  = int(len(X) * 0.8)
    X_tr, X_vl     = X.iloc[:split], X.iloc[split:]
    y_b_tr, y_b_vl = y_buy.iloc[:split], y_buy.iloc[split:]
    y_s_tr, y_s_vl = y_sell.iloc[:split], y_sell.iloc[split:]

    buy_spec  = GBMSpecialist("buy")
    sell_spec = GBMSpecialist("sell")

    buy_spec.fit(X_tr, y_b_tr, X_vl, y_b_vl)
    sell_spec.fit(X_tr, y_s_tr, X_vl, y_s_vl)

    buy_spec.save(MODELS_DIR)
    sell_spec.save(MODELS_DIR)

    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feat_cols, f)
    logger.info(f"Feature columns saved: {len(feat_cols)} features")

    logger.info("\nTop-10 BUY features:")
    logger.info(buy_spec.get_feature_importance().head(10).to_string())
    logger.info("\nTop-10 SELL features:")
    logger.info(sell_spec.get_feature_importance().head(10).to_string())

    return buy_spec, sell_spec


def step_validate(df_labeled, feat_cols):
    logger.info("="*60)
    logger.info("STEP 6: Walk-forward validation")
    logger.info("="*60)

    X      = df_labeled[feat_cols]
    y_buy  = df_labeled["buy_win"]
    y_sell = df_labeled["sell_win"]

    def model_trainer(X_tr, y_b_tr, y_s_tr, X_val, y_b_val, y_s_val):
        stack = ModelStack()
        stack.buy_specialist.fit(X_tr, y_b_tr, X_val, y_b_val)
        stack.sell_specialist.fit(X_tr, y_s_tr, X_val, y_s_val)
        return stack

    validator = WalkForwardValidator()
    return validator.run(
        df_labeled, X, y_buy, y_sell,
        model_trainer,
        buy_threshold  = 0.60,
        sell_threshold = 0.60
    )


def step_stress_test(df_labeled, feat_cols, buy_spec, sell_spec):
    logger.info("="*60)
    logger.info("STEP 7: Stress test")
    logger.info("="*60)

    import numpy as np
    X = df_labeled[feat_cols]
    split = int(len(X) * 0.8)
    X_te  = X.iloc[split:]
    df_te = df_labeled.iloc[split:]

    buy_probs  = np.asarray(buy_spec.predict_proba(X_te), dtype=float)
    sell_probs = np.asarray(sell_spec.predict_proba(X_te), dtype=float)

    stress = run_stress_test(df_te, buy_probs, sell_probs)
    logger.info(f"Baseline PF: {stress['baseline'].get('profit_factor'):.3f}")
    logger.info(f"Stressed PF: {stress['stressed'].get('profit_factor'):.3f}")
    return stress


def run_full_training(skip_validation: bool = False):
    logger.info("\n" + "="*60)
    logger.info(f"{SETUP_TAG} — TRAINING PIPELINE")
    logger.info("="*60 + "\n")

    if not initialize_mt5():
        raise RuntimeError("MT5 connection failed. Check terminal is open and logged in.")

    try:
        data_aligned = step_fetch_data()
        df_features, feat_cols = step_build_features(data_aligned)
        df_labeled = step_build_labels(df_features, data_aligned)
        step_fit_regime(data_aligned)
        buy_spec, sell_spec = step_train_specialists(df_labeled, feat_cols)

        if not skip_validation:
            step_validate(df_labeled, feat_cols)
            step_stress_test(df_labeled, feat_cols, buy_spec, sell_spec)

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE. Models saved to: " + MODELS_DIR)
        logger.info("="*60)
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    run_full_training(skip_validation=False)
