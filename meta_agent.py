"""
=============================================================================
META-AGENT
Combines model stack predictions, regime context, early detection signals,
and uncertainty quantification into a final BUY / SELL / NO_TRADE decision.
This is the gatekeeper — nothing trades without passing all its filters.
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

from config import (
    META_THRESHOLDS, META_MIN_PROB_EDGE,
    REGIME_ADAPTIVE_MIN_CONF, REGIME_ADAPTIVE_PROB_BOOST,
    DAILY_PROFIT_TARGET, DAILY_LOSS_LIMIT, MAX_POSITIONS,
    CONSECUTIVE_LOSS_REDUCTION, CONSECUTIVE_LOSS_REDUCE_AFTER,
    CONSECUTIVE_LOSS_REDUCE_BY, CONSECUTIVE_LOSS_REDUCE_MAX,
    CONSECUTIVE_WIN_INCREASE_AFTER, CONSECUTIVE_WIN_INCREASE_BY,
    DIRECTIONAL_LOSS_LIMIT_ENABLED, DIRECTIONAL_LOSS_LIMIT_PCT,
    ENSEMBLE_COHERENCE_CHECK, MIN_COHERENCE_EDGE,
)

logger = logging.getLogger("MetaAgent")


@dataclass
class TradeDecision:
    """Final decision output from the meta-agent."""
    action:         str    # "BUY", "SELL", "NO_TRADE"
    confidence:     float  # Final probability
    regime:         str
    reason:         str    # Human-readable decision rationale
    block_reasons:  list   # What blocked the trade (if any)
    buy_prob:       float = 0.0
    sell_prob:      float = 0.0


class MetaAgent:
    """
    The final decision layer.

    Decision logic:
    1. Check daily P&L limits (hard stop if hit)
    2. Check early detection blocks (shock / regime transition / max positions)
    3. Get regime-specific thresholds
    4. Check model agreement count
    5. Check uncertainty (ensemble disagreement)
    6. Apply early detection score boost/penalty
    7. Final probability threshold check
    8. Emit BUY / SELL / NO_TRADE
    """

    def __init__(self):
        self._last_buy_bar  = -999  # Cooldown tracking
        self._last_sell_bar = -999
        self._bar_counter   = 0
        self._session_trades: Dict[str, int] = {"BUY": 0, "SELL": 0}
        # P1: Consecutive streak tracking
        self._consecutive_wins: Dict[str, int]   = {"BUY": 0, "SELL": 0}
        self._consecutive_losses: Dict[str, int] = {"BUY": 0, "SELL": 0}
        # P7: Directional P&L per session
        self._directional_pnl: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

    def decide(
        self,
        model_output:   Dict[str, float],
        regime_result:  Dict[str, object],
        open_positions: pd.DataFrame,
        daily_pnl:      float,
        current_bar:    int = 0,
        cooldown_bars:  int = 3,
    ) -> TradeDecision:
        """
        Evaluate GBM model output against regime-aware thresholds.

        Simplified to match the validated pipeline exactly:
        - GBM buy_prob / sell_prob vs flat 0.60 threshold
        - No uncertainty gate (was never in validation)
        - No agreement gate (was never in validation)
        - No ED score adjustment (ED scores were never in validation)
        - Daily P&L limits and cooldown are retained as risk controls
        """
        self._bar_counter = current_bar
        blocks = []

        # ── GATE 1: Daily P&L hard limits ─────────────────────────────────
        if daily_pnl >= DAILY_PROFIT_TARGET:
            return TradeDecision(
                action="NO_TRADE", confidence=0.0,
                regime=str(regime_result.get("regime", "UNKNOWN")),
                reason="Daily profit target reached. Trading stopped.",
                block_reasons=["DAILY_PROFIT_TARGET"]
            )
        if daily_pnl <= DAILY_LOSS_LIMIT:
            return TradeDecision(
                action="NO_TRADE", confidence=0.0,
                regime=str(regime_result.get("regime", "UNKNOWN")),
                reason="Daily loss limit hit. Trading stopped.",
                block_reasons=["DAILY_LOSS_LIMIT"]
            )

        # ── GET REGIME + THRESHOLDS ────────────────────────────────────────
        regime     = str(regime_result.get("regime", "UNKNOWN"))
        reg_conf   = float(regime_result.get("confidence", 0.0))
        thresholds = META_THRESHOLDS.get(regime, META_THRESHOLDS["UNKNOWN"])
        buy_thresh  = thresholds["buy"]
        sell_thresh = thresholds["sell"]
        if reg_conf < REGIME_ADAPTIVE_MIN_CONF:
            buy_thresh  += REGIME_ADAPTIVE_PROB_BOOST
            sell_thresh += REGIME_ADAPTIVE_PROB_BOOST
            logger.info(f"[ADAPTIVE] conf={reg_conf:.2f} < {REGIME_ADAPTIVE_MIN_CONF} "
                        f"boosted thresholds to buy={buy_thresh:.2f} sell={sell_thresh:.2f}")

        # ── EXTRACT MODEL SIGNALS (GBM only) ──────────────────────────────
        buy_prob  = float(model_output.get("buy_prob",  0.0))
        sell_prob = float(model_output.get("sell_prob", 0.0))

        # ── P8: ENSEMBLE COHERENCE CHECK ──────────────────────────────────
        coherence_gate = True
        coherence_msg = ""
        if ENSEMBLE_COHERENCE_CHECK:
            if buy_prob >= buy_thresh and (buy_prob - sell_prob) < MIN_COHERENCE_EDGE:
                coherence_msg = f"coherence_buy: buy={buy_prob:.3f} sell={sell_prob:.3f} edge<{MIN_COHERENCE_EDGE}"
                coherence_gate = False
            elif sell_prob >= sell_thresh and (sell_prob - buy_prob) < MIN_COHERENCE_EDGE:
                coherence_msg = f"coherence_sell: sell={sell_prob:.3f} buy={buy_prob:.3f} edge<{MIN_COHERENCE_EDGE}"
                coherence_gate = False
        if not coherence_gate:
            return TradeDecision(
                action="NO_TRADE", confidence=max(buy_prob, sell_prob),
                regime=regime,
                reason="NO_TRADE: " + coherence_msg,
                block_reasons=["COHERENCE_GATE"],
                buy_prob=buy_prob, sell_prob=sell_prob,
            )

        # ── GATE 2: Open position limits ───────────────────────────────────
        n_buy_open  = 0
        n_sell_open = 0
        if not open_positions.empty:
            n_buy_open  = (open_positions["type"] == "BUY").sum()
            n_sell_open = (open_positions["type"] == "SELL").sum()

        # ── GATE 3: Cooldown ───────────────────────────────────────────────
        buy_on_cooldown  = (current_bar - self._last_buy_bar)  < cooldown_bars
        sell_on_cooldown = (current_bar - self._last_sell_bar) < cooldown_bars

        # ── P7: Directional loss block ─────────────────────────────────────
        for s in ("BUY", "SELL"):
            reason = self.directional_block(s, daily_pnl)
            if reason:
                blocks.append(reason)

        # ── DECISION ──────────────────────────────────────────────────────────
        buy_edge = buy_prob - sell_prob
        sell_edge = sell_prob - buy_prob

        can_buy  = (
            buy_prob  >= buy_thresh
            and buy_edge >= META_MIN_PROB_EDGE
            and not buy_on_cooldown
            and n_buy_open < MAX_POSITIONS
        )
        can_sell = (
            sell_prob >= sell_thresh
            and sell_edge >= META_MIN_PROB_EDGE
            and not sell_on_cooldown
            and n_sell_open < MAX_POSITIONS
        )

        # Mutual exclusion: pick the stronger signal
        if can_buy and can_sell:
            if buy_prob >= sell_prob:
                can_sell = False
            else:
                can_buy  = False

        if can_buy:
            return TradeDecision(
                action     = "BUY",
                confidence = buy_prob,
                regime     = regime,
                reason     = (
                    f"BUY | prob={buy_prob:.3f} >= thresh={buy_thresh} "
                    f"| edge={buy_edge:.3f} >= {META_MIN_PROB_EDGE} "
                    f"| regime={regime}({reg_conf:.2f})"
                ),
                block_reasons  = [],
                buy_prob       = buy_prob,
                sell_prob      = sell_prob,
            )

        if can_sell:
            return TradeDecision(
                action     = "SELL",
                confidence = sell_prob,
                regime     = regime,
                reason     = (
                    f"SELL | prob={sell_prob:.3f} >= thresh={sell_thresh} "
                    f"| edge={sell_edge:.3f} >= {META_MIN_PROB_EDGE} "
                    f"| regime={regime}({reg_conf:.2f})"
                ),
                block_reasons  = [],
                buy_prob       = buy_prob,
                sell_prob      = sell_prob,
            )

        # Build NO_TRADE reason
        reason_parts = []
        if buy_prob < buy_thresh:
            reason_parts.append(f"buy_prob={buy_prob:.3f} < {buy_thresh}")
        elif (buy_prob - sell_prob) < META_MIN_PROB_EDGE:
            reason_parts.append(f"buy_edge={(buy_prob - sell_prob):.3f} < {META_MIN_PROB_EDGE}")
        if sell_prob < sell_thresh:
            reason_parts.append(f"sell_prob={sell_prob:.3f} < {sell_thresh}")
        elif (sell_prob - buy_prob) < META_MIN_PROB_EDGE:
            reason_parts.append(f"sell_edge={(sell_prob - buy_prob):.3f} < {META_MIN_PROB_EDGE}")
        if buy_on_cooldown:
            reason_parts.append("BUY cooldown")
        if sell_on_cooldown:
            reason_parts.append("SELL cooldown")
        if n_buy_open >= MAX_POSITIONS:
            reason_parts.append(f"max BUY positions ({n_buy_open})")
        if n_sell_open >= MAX_POSITIONS:
            reason_parts.append(f"max SELL positions ({n_sell_open})")
        if blocks:
            reason_parts.extend(blocks)

        return TradeDecision(
            action        = "NO_TRADE",
            confidence    = max(buy_prob, sell_prob),
            regime        = regime,
            reason        = "NO_TRADE: " + " | ".join(reason_parts),
            block_reasons = [],
            buy_prob      = buy_prob,
            sell_prob     = sell_prob,
        )

    def record_executed_signal(self, action: str, current_bar: int):
        """
        Record cooldown only after real exposure is created.

        Important live fix:
        The previous version updated cooldown immediately when a BUY/SELL
        decision was emitted. That was too conservative for virtual-confirmation
        mode because an unfilled virtual entry could block later valid signals
        even though no trade or broker pending order existed.
        """
        action = str(action).upper()
        if action == "BUY":
            self._last_buy_bar = int(current_bar)
            self._session_trades["BUY"] += 1
        elif action == "SELL":
            self._last_sell_bar = int(current_bar)
            self._session_trades["SELL"] += 1

    # Backwards-compatible alias for callers.
    record_entry = record_executed_signal

    def record_trade_outcome(self, side: str, pnl: float):
        """
        Record a closed trade outcome for streak tracking (P1) and
        directional P&L tracking (P7).
        """
        side = str(side).upper()
        if side not in ("BUY", "SELL"):
            return
        self._directional_pnl[side] = self._directional_pnl.get(side, 0.0) + pnl
        if pnl > 0:
            self._consecutive_wins[side]   = self._consecutive_wins.get(side, 0) + 1
            self._consecutive_losses[side] = 0
        else:
            self._consecutive_losses[side] = self._consecutive_losses.get(side, 0) + 1
            self._consecutive_wins[side]   = 0

    def streak_multiplier(self, side: str) -> float:
        """
        P1: Return a multiplier based on consecutive win/loss streak.
        ≥CONSECUTIVE_LOSS_REDUCE_AFTER losses -> reduce sizing.
        ≥CONSECUTIVE_WIN_INCREASE_AFTER wins  -> increase sizing.
        """
        if not CONSECUTIVE_LOSS_REDUCTION:
            return 1.0
        side = str(side).upper()
        losses = self._consecutive_losses.get(side, 0)
        wins   = self._consecutive_wins.get(side, 0)
        if losses >= CONSECUTIVE_LOSS_REDUCE_AFTER:
            levels = min(losses - CONSECUTIVE_LOSS_REDUCE_AFTER + 1,
                         CONSECUTIVE_LOSS_REDUCE_MAX)
            return max(0.01, CONSECUTIVE_LOSS_REDUCE_BY ** levels)
        if wins >= CONSECUTIVE_WIN_INCREASE_AFTER:
            levels = wins - CONSECUTIVE_WIN_INCREASE_AFTER + 1
            return CONSECUTIVE_WIN_INCREASE_BY ** levels
        return 1.0

    def directional_block(self, side: str, daily_pnl: float) -> Optional[str]:
        """
        P7: Return a block reason if this side has consumed too much of the
        daily loss limit.
        """
        if not DIRECTIONAL_LOSS_LIMIT_ENABLED:
            return None
        side = str(side).upper()
        dir_pnl = self._directional_pnl.get(side, 0.0)
        limit = abs(DAILY_LOSS_LIMIT) * DIRECTIONAL_LOSS_LIMIT_PCT
        if dir_pnl <= -limit:
            return f"directional_loss_limit {side}: {dir_pnl:.2f} <= -{limit:.2f}"
        return None

    def reset_session_counters(self):
        """Call at start of each trading session."""
        self._session_trades = {"BUY": 0, "SELL": 0}
        self._last_buy_bar   = -999
        self._last_sell_bar  = -999
        self._consecutive_wins   = {"BUY": 0, "SELL": 0}
        self._consecutive_losses = {"BUY": 0, "SELL": 0}
        self._directional_pnl    = {"BUY": 0.0, "SELL": 0.0}

    def get_session_stats(self) -> Dict:
        return {
            "session_buys":  self._session_trades["BUY"],
            "session_sells": self._session_trades["SELL"],
            "last_buy_bar":  self._last_buy_bar,
            "last_sell_bar": self._last_sell_bar,
            "consecutive_wins":   dict(self._consecutive_wins),
            "consecutive_losses": dict(self._consecutive_losses),
            "directional_pnl":    dict(self._directional_pnl),
        }
