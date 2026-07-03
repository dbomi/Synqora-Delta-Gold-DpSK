"""
=============================================================================
RETRAIN HMM — SYNQORA DELTA GOLD FABLE
Refits ONLY the regime intelligence (5-state HMM + CUSUM) on fresh MT5
history and saves it into this setup's models directory. The shipped GBM
BUY/SELL specialists are untouched.

Usage:  python retrain_hmm.py
Requires the MT5 terminal to be open and logged in.
=============================================================================
"""

import logging
import os
from datetime import datetime

from config import (
    SETUP_TAG, SYMBOL, PRIMARY_TF, CONTEXT_TFS,
    TRAIN_START, TRAIN_END, MODELS_DIR, LOGS_DIR,
)
from data_engine import initialize_mt5, shutdown_mt5, fetch_multi_tf, align_to_primary
from regime_detector import RegimeRouter

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    handlers= [
        logging.FileHandler(os.path.join(LOGS_DIR, f"retrain_hmm_{datetime.now():%Y%m%d_%H%M%S}.log"),
                            mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RetrainHMM")


def retrain_hmm() -> RegimeRouter:
    logger.info("="*60)
    logger.info(f"{SETUP_TAG} — HMM REGIME RETRAIN")
    logger.info("="*60)

    if not initialize_mt5():
        raise RuntimeError("MT5 connection failed. Check terminal is open and logged in.")

    try:
        start = datetime.strptime(TRAIN_START, "%Y-%m-%d")
        end   = datetime.strptime(TRAIN_END,   "%Y-%m-%d")

        data = fetch_multi_tf(SYMBOL, PRIMARY_TF, CONTEXT_TFS, start, end, use_cache=True)
        if PRIMARY_TF not in data or data[PRIMARY_TF].empty:
            raise RuntimeError(f"Failed to fetch primary TF data ({PRIMARY_TF})")

        data_aligned = align_to_primary(data, PRIMARY_TF)
        df_primary   = data_aligned[PRIMARY_TF]
        logger.info(f"Fitting HMM on {len(df_primary)} {PRIMARY_TF} bars "
                    f"({df_primary.index[0]} – {df_primary.index[-1]})")

        router = RegimeRouter()
        router.fit(df_primary)
        router.save(MODELS_DIR)

        # Sanity check: report current regime from the freshly fitted router.
        current = router.get_regime(df_primary.tail(300))
        logger.info(f"Retrained HMM state map: {router.hmm.state_map}")
        logger.info(f"Current regime read-back: {current}")
        logger.info(f"Saved to {MODELS_DIR}")
        return router
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    retrain_hmm()
