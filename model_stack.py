"""
=============================================================================
MODEL STACK — SYNQORA DELTA GOLD FABLE
GBM BUY and SELL specialists only (LightGBM + XGBoost ensemble per side).

The LSTM temporal model and CatBoost meta-learner from the v7/delta lineage
are deliberately excluded: they were never part of the walk-forward
validated pipeline (PF=5.79, win_rate=72.2%, Sharpe=2.92), and the
meta-learner was trained on placeholder inputs. Fable runs exactly the
validated model set.
=============================================================================
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List

import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

from config import LGBM_PARAMS, XGB_PARAMS, MODELS_DIR

logger = logging.getLogger("ModelStack")


class GBMSpecialist:
    """
    Gradient boosting specialist for one side (BUY or SELL).
    Ensembles LightGBM and XGBoost, outputs the mean probability.
    """

    def __init__(self, side: str):
        assert side in ("buy", "sell"), "side must be 'buy' or 'sell'"
        self.side       = side
        self.lgbm       = None
        self.xgb_model  = None
        self.scaler     = StandardScaler()
        self.feature_cols: List[str] = []
        self.fitted     = False
        self._lgbm_best_iter = 200

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series
    ) -> "GBMSpecialist":
        self.feature_cols = list(X_train.columns)
        logger.info(f"[{self.side.upper()}] Training GBM specialist on {len(X_train)} samples "
                    f"({y_train.mean():.1%} positive rate)...")

        X_tr_sc = self.scaler.fit_transform(X_train)
        X_vl_sc = self.scaler.transform(X_val)

        # ── LightGBM ──
        params = {**LGBM_PARAMS}
        dtrain = lgb.Dataset(X_tr_sc, label=y_train)
        dval   = lgb.Dataset(X_vl_sc, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
        self.lgbm = lgb.train(
            params,
            dtrain,
            num_boost_round = LGBM_PARAMS["n_estimators"],
            valid_sets      = [dval],
            callbacks       = callbacks
        )
        self._lgbm_best_iter = self.lgbm.best_iteration

        # ── XGBoost ──
        xgb_params = {k: v for k, v in XGB_PARAMS.items() if k != "n_estimators"}
        self.xgb_model = xgb.XGBClassifier(
            **xgb_params,
            n_estimators = XGB_PARAMS["n_estimators"],
            early_stopping_rounds = 50,
        )
        self.xgb_model.fit(
            X_tr_sc, y_train,
            eval_set = [(X_vl_sc, y_val)],
            verbose  = False
        )

        self.fitted = True
        self._evaluate(X_vl_sc, y_val)
        return self

    def _evaluate(self, X_val_sc: np.ndarray, y_val: pd.Series):
        probs = self._predict_proba(X_val_sc)
        auc   = roc_auc_score(y_val, probs)
        preds = (probs >= 0.5).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        logger.info(f"[{self.side.upper()}] Val AUC={auc:.4f} | F1={f1:.4f}")

    def _predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        lgbm_prob = self.lgbm.predict(X_scaled)
        xgb_prob  = self.xgb_model.predict_proba(X_scaled)[:, 1]
        return (lgbm_prob + xgb_prob) / 2.0

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"{self.side} specialist not fitted.")
        X_aligned = X.reindex(columns=self.feature_cols, fill_value=0.0)
        X_scaled  = self.scaler.transform(X_aligned)
        return self._predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.Series:
        lgbm_imp = pd.Series(
            self.lgbm.feature_importance(importance_type="gain"),
            index=self.feature_cols
        )
        return lgbm_imp.sort_values(ascending=False)

    def save(self, directory: str = MODELS_DIR):
        os.makedirs(directory, exist_ok=True)
        state = {
            "lgbm":         self.lgbm,
            "xgb":          self.xgb_model,
            "scaler":       self.scaler,
            "feature_cols": self.feature_cols,
            "lgbm_best":    self._lgbm_best_iter,
        }
        with open(os.path.join(directory, f"gbm_{self.side}.pkl"), "wb") as f:
            pickle.dump(state, f)
        logger.info(f"[{self.side.upper()}] GBM specialist saved.")

    def load(self, directory: str = MODELS_DIR) -> "GBMSpecialist":
        path = os.path.join(directory, f"gbm_{self.side}.pkl")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.lgbm          = state["lgbm"]
        self.xgb_model     = state["xgb"]
        self.scaler        = state["scaler"]
        self.feature_cols  = state["feature_cols"]
        self._lgbm_best_iter = state.get("lgbm_best", 200)
        self.fitted        = True
        logger.info(f"[{self.side.upper()}] GBM specialist loaded from {path}")
        return self


class ModelStack:
    """
    Holds the two GBM specialists. Unified predict interface for the
    live trader and trainer.
    """

    def __init__(self):
        self.buy_specialist  = GBMSpecialist("buy")
        self.sell_specialist = GBMSpecialist("sell")
        self.trained         = False

    def predict(self, X_features: pd.DataFrame) -> Dict[str, float]:
        """BUY and SELL probabilities for the most recent bar."""
        last_row  = X_features.tail(1)
        buy_prob  = float(self.buy_specialist.predict_proba(last_row)[0])
        sell_prob = float(self.sell_specialist.predict_proba(last_row)[0])
        return {
            "buy_prob":  buy_prob,
            "sell_prob": sell_prob,
        }

    def save(self, directory: str = MODELS_DIR):
        self.buy_specialist.save(directory)
        self.sell_specialist.save(directory)
        logger.info("ModelStack (GBM specialists) saved.")

    def load(self, directory: str = MODELS_DIR) -> "ModelStack":
        self.buy_specialist.load(directory)
        self.sell_specialist.load(directory)
        self.trained = True
        logger.info("ModelStack (GBM specialists) loaded.")
        return self
