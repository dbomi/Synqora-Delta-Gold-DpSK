"""
=============================================================================
SIGNAL QUEUE GATE — SYNQORA DELTA GOLD FABLE
Model signals are never executed directly. They enter a bounded FIFO queue
and are released only when short-term LEADING indicators confirm the move
is starting — scored on M1 resolution each M1 cycle.

Queue mechanics:
  - Capacity 20; when full the OLDEST signal is evicted to make room.
  - BUY/SELL signals from all families coexist.
  - Each slot stores: side, family, source_cid, queue_price, queue_time,
    m15_atr (plus queue-time spread for the spread-tightening check).
  - Expiry: QUEUE_MAX_PENDING_MINUTES — signals don't live forever.
  - Release: up to QUEUE_MAX_RELEASES_PER_CYCLE per M1 cycle,
    max QUEUE_MAX_RELEASES_PER_SIDE per side per cycle (direction-balanced).
  - Release threshold: score ≥ QUEUE_RELEASE_SCORE (≈2-3 independent
    confirmations).

Hard blocks (news blackout / intraday extreme / H4 topzone) are evaluated
by the caller via entry_guards and passed in as a callable so this module
stays broker-free and unit-testable.
=============================================================================
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    QUEUE_CAPACITY, QUEUE_MAX_PENDING_MINUTES, QUEUE_RELEASE_SCORE,
    QUEUE_MAX_RELEASES_PER_CYCLE, QUEUE_MAX_RELEASES_PER_SIDE,
    QUEUE_SCORE_WEIGHTS, QUEUE_M1_ROC_PERIOD, QUEUE_VOL_SPIKE_MULT,
    QUEUE_VOL_AVG_BARS, QUEUE_BODY_MIN_PROPORTION, QUEUE_WICK_BODY_RATIO,
    QUEUE_SPREAD_TIGHTEN_RATIO,
)

logger = logging.getLogger("SignalQueue")


# ─────────────────────────────────────────────────────────────────────────────
# QUEUED SIGNAL SLOT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueuedSignal:
    side:         str                 # "BUY" | "SELL"
    family:       str                 # signal family, e.g. "GBM_M15"
    source_cid:   str                 # unique signal id
    queue_price:  float               # price when the signal was queued
    queue_time:   datetime            # UTC time when queued
    m15_atr:      float               # M15 ATR at queue time (SL/TP geometry)
    queue_spread: float = 0.0         # spread (points) at queue time
    meta:         Dict = field(default_factory=dict)   # probs, regime, etc.

    def age_minutes(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        qt  = self.queue_time
        if qt.tzinfo is None:
            qt = qt.replace(tzinfo=timezone.utc)
        return (now - qt).total_seconds() / 60.0


# ─────────────────────────────────────────────────────────────────────────────
# LEADING-INDICATOR SCORER (M1 resolution)
# ─────────────────────────────────────────────────────────────────────────────

def _direction_sign(side: str) -> int:
    return 1 if str(side).upper() == "BUY" else -1


def score_signal(
    sig:            QueuedSignal,
    df_m1:          pd.DataFrame,
    df_m5:          Optional[pd.DataFrame],
    current_price:  float,
    current_spread: float,
) -> Dict:
    """
    Score one queued signal against genuinely-leading M1 evidence.

    df_m1 / df_m5 must contain only CLOSED bars (caller drops the forming
    bar), newest last, with open/high/low/close/tick_volume columns.

    Returns {"score": float, "components": {name: contribution}}.
    """
    comp: Dict[str, float] = {}
    W    = QUEUE_SCORE_WEIGHTS
    sign = _direction_sign(sig.side)

    if df_m1 is None or len(df_m1) < max(QUEUE_VOL_AVG_BARS, QUEUE_M1_ROC_PERIOD * 3) + 2:
        return {"score": 0.0, "components": {}, "reason": "insufficient_m1_data"}

    closes = df_m1["close"].astype(float)
    last   = df_m1.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    rng    = max(h - l, 1e-9)
    body   = abs(c - o)
    bar_dir = np.sign(c - o)

    # 1. M1 momentum zero-cross — ROC turns positive (BUY) / negative (SELL)
    #    on the last closed bar. Catches the turn before price moves.
    roc      = closes.pct_change(QUEUE_M1_ROC_PERIOD)
    roc_now  = float(roc.iloc[-1])
    roc_prev = float(roc.iloc[-2])
    if np.isfinite(roc_now) and np.isfinite(roc_prev):
        if sign > 0 and roc_prev <= 0.0 < roc_now:
            comp["m1_momentum_zero_cross"] = W["m1_momentum_zero_cross"]
        elif sign < 0 and roc_prev >= 0.0 > roc_now:
            comp["m1_momentum_zero_cross"] = W["m1_momentum_zero_cross"]

    # 2. Tick volume spike in direction — last closed M1 tick volume above
    #    QUEUE_VOL_SPIKE_MULT × rolling average AND candle closed in direction.
    vol = df_m1["tick_volume"].astype(float)
    vol_avg = float(vol.iloc[-(QUEUE_VOL_AVG_BARS + 1):-1].mean())
    if vol_avg > 0 and float(vol.iloc[-1]) > QUEUE_VOL_SPIKE_MULT * vol_avg and bar_dir == sign:
        comp["tick_volume_spike"] = W["tick_volume_spike"]

    # 3. Candle body proportion — conviction bar in the signal direction.
    if body >= QUEUE_BODY_MIN_PROPORTION * rng and bar_dir == sign:
        comp["candle_body_proportion"] = W["candle_body_proportion"]

    # 4. Pullback to or beyond queue price — better entry than at signal time.
    if sign > 0 and current_price <= sig.queue_price:
        comp["pullback_to_queue_price"] = W["pullback_to_queue_price"]
    elif sign < 0 and current_price >= sig.queue_price:
        comp["pullback_to_queue_price"] = W["pullback_to_queue_price"]

    # 5. Price rejection wick — lower wick (BUY) / upper wick (SELL)
    #    at least QUEUE_WICK_BODY_RATIO × body.
    body_top    = max(o, c)
    body_bottom = min(o, c)
    lower_wick  = body_bottom - l
    upper_wick  = h - body_top
    wick = lower_wick if sign > 0 else upper_wick
    if wick >= QUEUE_WICK_BODY_RATIO * max(body, 1e-9):
        comp["rejection_wick"] = W["rejection_wick"]

    # 6. M1 ROC acceleration — momentum gaining, not just present.
    roc_accel = roc.diff()
    accel_now = float(roc_accel.iloc[-1])
    if np.isfinite(accel_now) and sign * accel_now > 0:
        comp["roc_acceleration"] = W["roc_acceleration"]

    # 7. M5 alignment — last closed M5 candle in the signal direction.
    if df_m5 is not None and len(df_m5) >= 1:
        m5 = df_m5.iloc[-1]
        m5_dir = np.sign(float(m5["close"]) - float(m5["open"]))
        if m5_dir == sign:
            comp["m5_alignment"] = W["m5_alignment"]

    # 8. Spread tightening — liquidity improving vs queue time.
    if sig.queue_spread > 0 and current_spread > 0:
        if current_spread <= QUEUE_SPREAD_TIGHTEN_RATIO * sig.queue_spread:
            comp["spread_tightening"] = W["spread_tightening"]

    return {"score": float(sum(comp.values())), "components": comp}


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL QUEUE
# ─────────────────────────────────────────────────────────────────────────────

class SignalQueue:
    """
    Bounded FIFO signal queue with leading-indicator gated release.
    """

    def __init__(self, capacity: int = QUEUE_CAPACITY):
        self.capacity = int(capacity)
        self._slots: deque[QueuedSignal] = deque()

    # ── Introspection ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._slots)

    def snapshot(self) -> List[Dict]:
        return [
            {
                "side": s.side, "family": s.family, "cid": s.source_cid,
                "queue_price": s.queue_price,
                "queue_time": s.queue_time.isoformat(),
                "m15_atr": s.m15_atr,
                "age_min": round(s.age_minutes(), 1),
            }
            for s in self._slots
        ]

    def has_recent_same_signal(self, side: str, family: str, within_minutes: float) -> bool:
        side = str(side).upper()
        for s in self._slots:
            if s.side == side and s.family == family and s.age_minutes() <= within_minutes:
                return True
        return False

    # ── Enqueue ────────────────────────────────────────────────────────────
    def enqueue(self, sig: QueuedSignal) -> Optional[QueuedSignal]:
        """
        Add a signal. If the queue is full, evict the oldest (first in)
        to make room. Returns the evicted signal, if any.
        """
        evicted = None
        if len(self._slots) >= self.capacity:
            evicted = self._slots.popleft()
            logger.info(f"[QUEUE] Full ({self.capacity}). Evicted oldest: "
                        f"{evicted.side} {evicted.source_cid} age={evicted.age_minutes():.1f}min")
        self._slots.append(sig)
        logger.info(f"[QUEUE] Enqueued {sig.side} {sig.family} cid={sig.source_cid} "
                    f"price={sig.queue_price:.2f} atr={sig.m15_atr:.2f} "
                    f"(depth={len(self._slots)}/{self.capacity})")
        return evicted

    # ── Expiry ─────────────────────────────────────────────────────────────
    def expire_stale(self, max_pending_minutes: float = QUEUE_MAX_PENDING_MINUTES,
                     now: Optional[datetime] = None) -> List[QueuedSignal]:
        """Drop signals older than max_pending_minutes. Returns dropped list."""
        expired = [s for s in self._slots if s.age_minutes(now) > max_pending_minutes]
        for s in expired:
            self._slots.remove(s)
            logger.info(f"[QUEUE] Expired {s.side} {s.source_cid} "
                        f"age={s.age_minutes(now):.1f}min > {max_pending_minutes}min")
        return expired

    # ── Release cycle (call once per closed M1 bar) ────────────────────────
    def release_cycle(
        self,
        df_m1:          pd.DataFrame,
        df_m5:          Optional[pd.DataFrame],
        current_price:  float,
        current_spread: float,
        hard_block_fn:  Optional[Callable[[QueuedSignal], Optional[str]]] = None,
        now:            Optional[datetime] = None,
    ) -> List[Dict]:
        """
        One M1 release cycle:
          1. Expire stale signals.
          2. Score every queued signal with the leading-indicator set.
          3. Filter score ≥ QUEUE_RELEASE_SCORE and not hard-blocked.
          4. Release best-first, capped at QUEUE_MAX_RELEASES_PER_CYCLE total
             and QUEUE_MAX_RELEASES_PER_SIDE per side.

        hard_block_fn(sig) returns a block-reason string (blocked) or None.
        Hard-blocked signals STAY in the queue (they may release later),
        they are just not released this cycle.

        Returns list of {"signal": QueuedSignal, "score": float,
                         "components": dict} released this cycle.
        """
        self.expire_stale(now=now)
        if not self._slots:
            return []

        candidates = []
        for sig in list(self._slots):
            result = score_signal(sig, df_m1, df_m5, current_price, current_spread)
            score  = result["score"]
            if score < QUEUE_RELEASE_SCORE:
                continue

            if hard_block_fn is not None:
                block_reason = hard_block_fn(sig)
                if block_reason:
                    logger.info(f"[QUEUE] {sig.side} {sig.source_cid} score={score:.1f} "
                                f"HARD-BLOCKED: {block_reason}")
                    continue

            candidates.append({"signal": sig, "score": score,
                               "components": result["components"]})

        if not candidates:
            return []

        # Best-first release, direction-balanced caps.
        candidates.sort(key=lambda x: x["score"], reverse=True)
        released: List[Dict] = []
        per_side = {"BUY": 0, "SELL": 0}

        for cand in candidates:
            if len(released) >= QUEUE_MAX_RELEASES_PER_CYCLE:
                break
            side = cand["signal"].side
            if per_side[side] >= QUEUE_MAX_RELEASES_PER_SIDE:
                continue
            per_side[side] += 1
            released.append(cand)
            self._slots.remove(cand["signal"])
            comp_str = ", ".join(f"{k}=+{v:.1f}" for k, v in cand["components"].items())
            logger.info(f"[QUEUE] RELEASE {side} {cand['signal'].source_cid} "
                        f"score={cand['score']:.1f} [{comp_str}]")

        return released
