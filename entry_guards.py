"""
=============================================================================
ENTRY GUARDS — SYNQORA DELTA GOLD FABLE
Hard blocks that prevent a queued signal from releasing regardless of its
leading-indicator score:

  1. News blackout window active.
  2. Intraday extreme guard — entry still within 1.5× ATR(M15) of the
     session high (BUY) / session low (SELL): chasing the extreme.
  3. H4 topzone guard — BUY in the top zone of the rolling H4 range
     (mirrored bottomzone for SELL).

All functions return a block-reason string when the guard fires, or None
when the entry is allowed. They are pure data-in/data-out so they can be
unit-tested without MT5.
=============================================================================
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    NEWS_BLACKOUT_ENABLED, NEWS_STATIC_WINDOWS_UTC, NEWS_EVENTS_FILE,
    NEWS_PRE_EVENT_MINUTES, NEWS_POST_EVENT_MINUTES,
    INTRADAY_EXTREME_GUARD_ENABLED, INTRADAY_EXTREME_ATR_MULT,
    H4_TOPZONE_GUARD_ENABLED, H4_TOPZONE_LOOKBACK_BARS, H4_TOPZONE_UPPER_PCT,
    H4_BOTTOMZONE_GUARD_ENABLED, H4_BOTTOMZONE_LOWER_PCT,
    GUARD_TREND_REGIME_EXEMPTION, GUARD_EXEMPT_MIN_REGIME_CONF,
    SESSION_OPEN_BLOCK_ENABLED, SESSION_OPEN_BLOCK_MINUTES,
    SESSION_OPEN_BLOCK_SESSIONS,
    EXHAUSTION_FILTER_ENABLED, EXHAUSTION_MIN_CONSECUTIVE, EXHAUSTION_ROC_BARS,
    REGIME_DIRECTION_GATE_ENABLED, REGIME_DIRECTION_GATE_CONFIDENCE,
    REGIME_DIRECTION_GATE_BLOCK_SELL_UP, REGIME_DIRECTION_GATE_BLOCK_BUY_DN,
    REGIME_DIRECTION_GATE_BLOCK_BUY_DN_MIN_CONF,
    REGIME_CONFLICT_RESOLVER_ENABLED, REGIME_CONFLICT_OVERRIDE_MIN_PROB,
    REGIME_CONFLICT_OVERRIDE_MIN_EDGE, REGIME_CONFLICT_OVERRIDE_MIN_SCORE,
    REGIME_CONFLICT_REDUCED_RISK_MULT, REGIME_CONFLICT_STRUCTURE_BARS,
    REGIME_CONFLICT_EMA_FAST, REGIME_CONFLICT_EMA_SLOW,
    REGIME_CONFLICT_MIN_CONTRADICTIONS, REGIME_CONFLICT_HARD_BLOCK_MIN_SUPPORT,
    MOMENTUM_ALIGNMENT_ENABLED, MOMENTUM_ALIGNMENT_LOOKBACK,
    MOMENTUM_ALIGNMENT_MIN_POINTS,
)

logger = logging.getLogger("EntryGuards")


# ─────────────────────────────────────────────────────────────────────────────
# 1. NEWS BLACKOUT
# ─────────────────────────────────────────────────────────────────────────────

_news_events_cache: Optional[List[datetime]] = None
_news_events_mtime: float = -1.0


def _load_news_events() -> List[datetime]:
    """
    Load event timestamps from NEWS_EVENTS_FILE (JSON list of objects with
    'time_utc' in ISO format). Cached; reloaded when the file changes so the
    calendar can be updated without restarting the trader.
    """
    global _news_events_cache, _news_events_mtime

    path = NEWS_EVENTS_FILE
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    if not os.path.exists(path):
        return []

    mtime = os.path.getmtime(path)
    if _news_events_cache is not None and mtime == _news_events_mtime:
        return _news_events_cache

    events: List[datetime] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            ts = item.get("time_utc") if isinstance(item, dict) else item
            dt = datetime.fromisoformat(str(ts))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            events.append(dt)
        logger.info(f"[NEWS] Loaded {len(events)} events from {path}")
    except Exception as e:
        logger.warning(f"[NEWS] Failed to parse {path}: {e}")

    _news_events_cache = events
    _news_events_mtime = mtime
    return events


def news_blackout_reason(now: Optional[datetime] = None) -> Optional[str]:
    """Return a reason string if a news blackout window is active."""
    if not NEWS_BLACKOUT_ENABLED:
        return None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Static daily windows (UTC "HH:MM" pairs; may wrap midnight)
    hm = now.hour * 60 + now.minute
    for start_s, end_s in NEWS_STATIC_WINDOWS_UTC:
        sh, sm = map(int, str(start_s).split(":"))
        eh, em = map(int, str(end_s).split(":"))
        start, end = sh * 60 + sm, eh * 60 + em
        active = (start <= hm < end) if start <= end else (hm >= start or hm < end)
        if active:
            return f"news_static_window {start_s}-{end_s} UTC"

    # Event calendar
    pre  = timedelta(minutes=NEWS_PRE_EVENT_MINUTES)
    post = timedelta(minutes=NEWS_POST_EVENT_MINUTES)
    for evt in _load_news_events():
        if evt - pre <= now <= evt + post:
            return f"news_event_blackout event={evt.isoformat()} " \
                   f"window=-{NEWS_PRE_EVENT_MINUTES}/+{NEWS_POST_EVENT_MINUTES}min"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTRADAY EXTREME GUARD
# ─────────────────────────────────────────────────────────────────────────────

def intraday_extreme_reason(
    side:          str,
    df_primary:    pd.DataFrame,
    current_price: float,
    m15_atr:       float,
    now:           Optional[datetime] = None,
) -> Optional[str]:
    """
    Block when the entry would still chase the session extreme:
      BUY  blocked while price is within INTRADAY_EXTREME_ATR_MULT × ATR
           of the session HIGH.
      SELL blocked while price is within the same distance of the session LOW.

    Session = current UTC trading day, computed from df_primary (M15 bars).
    """
    if not INTRADAY_EXTREME_GUARD_ENABLED:
        return None
    if df_primary is None or df_primary.empty or m15_atr <= 0:
        return None

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    idx = df_primary.index
    if idx.tz is None:
        session_mask = idx >= pd.Timestamp(now.date())
    else:
        session_mask = idx >= pd.Timestamp(now.date(), tz="UTC")
    session = df_primary.loc[session_mask]
    if session.empty:
        session = df_primary.tail(1)

    session_high = float(session["high"].max())
    session_low  = float(session["low"].min())
    limit = INTRADAY_EXTREME_ATR_MULT * float(m15_atr)

    side = str(side).upper()
    if side == "BUY":
        dist = session_high - float(current_price)
        if dist <= limit:
            return (f"intraday_extreme BUY within {dist:.2f} of session high "
                    f"{session_high:.2f} (limit {limit:.2f} = "
                    f"{INTRADAY_EXTREME_ATR_MULT}×ATR {m15_atr:.2f})")
    elif side == "SELL":
        dist = float(current_price) - session_low
        if dist <= limit:
            return (f"intraday_extreme SELL within {dist:.2f} of session low "
                    f"{session_low:.2f} (limit {limit:.2f} = "
                    f"{INTRADAY_EXTREME_ATR_MULT}×ATR {m15_atr:.2f})")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. H4 TOPZONE / BOTTOMZONE GUARD
# ─────────────────────────────────────────────────────────────────────────────

def h4_zone_reason(
    side:          str,
    df_h4:         pd.DataFrame,
    current_price: float,
) -> Optional[str]:
    """
    Block BUY when price sits in the top H4_TOPZONE_UPPER_PCT..1.0 zone of
    the rolling H4 range (last H4_TOPZONE_LOOKBACK_BARS closed H4 bars);
    mirrored bottomzone for SELL.
    """
    if df_h4 is None or len(df_h4) < 3:
        return None

    window = df_h4.tail(H4_TOPZONE_LOOKBACK_BARS)
    h4_high = float(window["high"].max())
    h4_low  = float(window["low"].min())
    rng     = h4_high - h4_low
    if rng <= 0:
        return None

    pos = (float(current_price) - h4_low) / rng
    side = str(side).upper()

    if side == "BUY" and H4_TOPZONE_GUARD_ENABLED and pos >= H4_TOPZONE_UPPER_PCT:
        return (f"h4_topzone BUY pos={pos:.2f} >= {H4_TOPZONE_UPPER_PCT} "
                f"(range {h4_low:.2f}-{h4_high:.2f}, {len(window)} H4 bars)")
    if side == "SELL" and H4_BOTTOMZONE_GUARD_ENABLED and pos <= H4_BOTTOMZONE_LOWER_PCT:
        return (f"h4_bottomzone SELL pos={pos:.2f} <= {H4_BOTTOMZONE_LOWER_PCT} "
                f"(range {h4_low:.2f}-{h4_high:.2f}, {len(window)} H4 bars)")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION OPEN BLOCK (P6)
# ─────────────────────────────────────────────────────────────────────────────

SESSION_TIMES = {
    "SYDNEY":   (21, 6),
    "TOKYO":    (0, 9),
    "LONDON":   (7, 16),
    "NEW_YORK": (12, 21),
}


def session_open_block_reason(now: Optional[datetime] = None) -> Optional[str]:
    """
    Block trades during the opening period of configured sessions.
    The first SESSION_OPEN_BLOCK_MINUTES of each selected session are blocked.
    """
    if not SESSION_OPEN_BLOCK_ENABLED:
        return None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    hm = now.hour * 60 + now.minute
    for sess_name in SESSION_OPEN_BLOCK_SESSIONS:
        start_h, end_h = SESSION_TIMES.get(sess_name, (0, 24))
        start_min = start_h * 60
        # Handle overnight sessions (e.g. SYDNEY 21:00-06:00 UTC)
        if start_h <= end_h:
            in_session = start_min <= hm < end_h * 60
        else:
            in_session = hm >= start_min or hm < end_h * 60
        if not in_session:
            continue
        session_elapsed = (hm - start_min) % 1440
        if session_elapsed <= SESSION_OPEN_BLOCK_MINUTES:
            return (f"session_open_block {sess_name} opened "
                    f"{session_elapsed}min ago < {SESSION_OPEN_BLOCK_MINUTES}min")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXHAUSTION + DIVERGENCE FILTER (M5 candle pattern guard)
# ─────────────────────────────────────────────────────────────────────────────


def exhaustion_divergence_reason(
    side:   str,
    df_m5:  pd.DataFrame,
) -> Optional[str]:
    """
    BUY-ONLY: block entries that chase exhausted moves on M5.
    BUY blocked when EXHAUSTION_MIN_CONSECUTIVE consecutive bearish M5
    candles AND M5 ROC3 < 0 (buying a falling knife).

    SELL side was tested live and found to kill more winners than it saves
    (valid bounce shorts flagged as exhaustion). BUY-only catches 17% of
    replay losers with 0% false positives on big winners.

    Returns a reason string if blocked, or None if allowed.
    """
    if not EXHAUSTION_FILTER_ENABLED:
        return None
    min_bars = max(EXHAUSTION_ROC_BARS, EXHAUSTION_MIN_CONSECUTIVE) + 3
    if df_m5 is None or len(df_m5) < min_bars:
        return None

    closes = df_m5["close"].astype(float)
    opens  = df_m5["open"].astype(float)

    # M5 ROC3 — is short-term momentum aligned with the entry?
    roc3 = float(closes.pct_change(EXHAUSTION_ROC_BARS).iloc[-1])
    if not np.isfinite(roc3):
        return None

    # Count consecutive same-direction candles (last EXHAUSTION_MIN_CONSECUTIVE bars)
    cons = 0
    for i in range(EXHAUSTION_MIN_CONSECUTIVE):
        idx = -1 - i
        if idx < -len(closes):
            return None  # not enough bars
        c_dir = np.sign(float(closes.iloc[idx]) - float(opens.iloc[idx]))
        if c_dir == 0:
            return None  # doji resets the count, don't block
        if i == 0:
            ref_dir = c_dir
        elif c_dir != ref_dir:
            return None  # streak broken, not exhaustion

    side = str(side).upper()

    if side == "BUY" and ref_dir < 0 and roc3 < 0:
        return (f"exhaustion_divergence BUY {EXHAUSTION_MIN_CONSECUTIVE}+ bearish "
                f"M5 candles (ref_dir={ref_dir}) + ROC3={roc3:+.4f} — falling knife")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. REGIME DIRECTION GATE (counter-trend blind spot guard)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RegimeDirectionAssessment:
    """Structured result for the conflict-aware HMM direction resolver."""
    action: str = "ALLOW"           # ALLOW | REDUCED | BLOCK
    reason: str = ""
    risk_multiplier: float = 1.0
    contradicted: bool = False
    evidence: Dict[str, object] = field(default_factory=dict)


def _sign(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _structure_direction(df: Optional[pd.DataFrame], bars: int) -> int:
    if df is None or len(df) < bars + 1 or "close" not in df:
        return 0
    closes = df["close"].astype(float)
    return _sign(float(closes.iloc[-1] - closes.iloc[-1 - bars]))


def _ema_direction(df: Optional[pd.DataFrame], fast: int, slow: int) -> int:
    if df is None or len(df) < slow + 2 or "close" not in df:
        return 0
    closes = df["close"].astype(float)
    ef = float(closes.ewm(span=fast, adjust=False).mean().iloc[-1])
    es = float(closes.ewm(span=slow, adjust=False).mean().iloc[-1])
    return _sign(ef - es)


def _last_candle_direction(df: Optional[pd.DataFrame]) -> int:
    if df is None or len(df) < 1:
        return 0
    row = df.iloc[-1]
    return _sign(float(row["close"]) - float(row["open"]))


def _break_of_structure_direction(df: Optional[pd.DataFrame], lookback: int = 3) -> int:
    if df is None or len(df) < lookback + 1:
        return 0
    latest = df.iloc[-1]
    prior = df.iloc[-(lookback + 1):-1]
    close = float(latest["close"])
    if close > float(prior["high"].max()):
        return 1
    if close < float(prior["low"].min()):
        return -1
    return 0


def _m1_slope_direction(df_m1: Optional[pd.DataFrame], bars: int = 3) -> int:
    return _structure_direction(df_m1, max(2, bars))


def assess_regime_direction(
    side: str,
    regime: Optional[dict],
    *,
    df_m15: Optional[pd.DataFrame] = None,
    df_h1: Optional[pd.DataFrame] = None,
    df_m5: Optional[pd.DataFrame] = None,
    df_m1: Optional[pd.DataFrame] = None,
    prob: float = 0.0,
    edge: float = 0.0,
    queue_score: float = 0.0,
) -> RegimeDirectionAssessment:
    if not REGIME_DIRECTION_GATE_ENABLED or not regime:
        return RegimeDirectionAssessment()

    side = str(side).upper()
    side_sign = 1 if side == "BUY" else -1
    rname = str(regime.get("regime", "")).upper()
    conf = float(regime.get("confidence", 0.0))
    hmm_sign = 1 if rname == "TREND_UP" else -1 if rname == "TREND_DOWN" else 0

    conflicts = (
        side == "SELL" and REGIME_DIRECTION_GATE_BLOCK_SELL_UP and rname == "TREND_UP"
    ) or (
        side == "BUY" and REGIME_DIRECTION_GATE_BLOCK_BUY_DN and rname == "TREND_DOWN"
        and conf >= REGIME_DIRECTION_GATE_BLOCK_BUY_DN_MIN_CONF
    )
    if not conflicts or conf < REGIME_DIRECTION_GATE_CONFIDENCE:
        return RegimeDirectionAssessment()

    if not REGIME_CONFLICT_RESOLVER_ENABLED:
        return RegimeDirectionAssessment(
            action="BLOCK",
            reason=f"regime_direction {side} in {rname} (conf={conf:.2f})",
        )

    bars = max(2, int(REGIME_CONFLICT_STRUCTURE_BARS))
    directions = {
        "m15_structure": _structure_direction(df_m15, bars),
        "m15_ema": _ema_direction(df_m15, REGIME_CONFLICT_EMA_FAST, REGIME_CONFLICT_EMA_SLOW),
        "h1_structure": _structure_direction(df_h1, bars),
        "h1_ema": _ema_direction(df_h1, REGIME_CONFLICT_EMA_FAST, REGIME_CONFLICT_EMA_SLOW),
        "m5_candle": _last_candle_direction(df_m5),
        "m5_bos": _break_of_structure_direction(df_m5, bars),
        "m1_slope": _m1_slope_direction(df_m1, bars),
    }
    structural_keys = ("m15_structure", "m15_ema", "h1_structure", "h1_ema", "m5_bos")
    contradiction_count = sum(directions[k] == side_sign for k in structural_keys)
    hmm_support_count = sum(directions[k] == hmm_sign for k in structural_keys)
    signal_support_count = sum(v == side_sign for v in directions.values())
    m5_aligned = directions["m5_candle"] == side_sign or directions["m5_bos"] == side_sign
    m1_aligned = directions["m1_slope"] == side_sign
    contradicted = contradiction_count >= REGIME_CONFLICT_MIN_CONTRADICTIONS
    high_quality = (
        float(prob) >= REGIME_CONFLICT_OVERRIDE_MIN_PROB
        and float(edge) >= REGIME_CONFLICT_OVERRIDE_MIN_EDGE
        and float(queue_score) >= REGIME_CONFLICT_OVERRIDE_MIN_SCORE
    )

    evidence = {
        **directions,
        "hmm_regime": rname,
        "hmm_confidence": conf,
        "hmm_support_count": hmm_support_count,
        "signal_support_count": signal_support_count,
        "contradiction_count": contradiction_count,
        "high_quality": high_quality,
        "m5_aligned": m5_aligned,
        "m1_aligned": m1_aligned,
    }

    if high_quality and m5_aligned and m1_aligned:
        return RegimeDirectionAssessment(
            action="REDUCED",
            risk_multiplier=REGIME_CONFLICT_REDUCED_RISK_MULT,
            contradicted=contradicted,
            evidence=evidence,
            reason=(f"regime conflict overridden at reduced risk: {side} vs {rname} "
                    f"conf={conf:.2f}, prob={prob:.3f}, edge={edge:.3f}, "
                    f"score={queue_score:.1f}, structure={contradiction_count}, "
                    f"M5/M1 aligned"),
        )

    if contradicted and float(prob) >= 0.70 and float(edge) >= 0.35 \
            and float(queue_score) >= 4.0 and m1_aligned and signal_support_count >= 3:
        return RegimeDirectionAssessment(
            action="REDUCED",
            risk_multiplier=REGIME_CONFLICT_REDUCED_RISK_MULT,
            contradicted=True,
            evidence=evidence,
            reason=(f"stale/contradicted HMM reduced-risk release: {side} vs {rname} "
                    f"conf={conf:.2f}, signal_support={signal_support_count}, "
                    f"hmm_support={hmm_support_count}"),
        )

    if hmm_support_count >= REGIME_CONFLICT_HARD_BLOCK_MIN_SUPPORT and not contradicted:
        reason = (f"regime_direction {side} in {rname} (conf={conf:.2f}) — "
                  f"HMM confirmed by {hmm_support_count} structure checks")
    else:
        reason = (f"regime_direction {side} in {rname} (conf={conf:.2f}) — "
                  f"override evidence insufficient: prob={prob:.3f}, edge={edge:.3f}, "
                  f"score={queue_score:.1f}, signal_support={signal_support_count}, "
                  f"hmm_support={hmm_support_count}")
    return RegimeDirectionAssessment(
        action="BLOCK", reason=reason, contradicted=contradicted, evidence=evidence,
    )


def regime_direction_reason(side: str, regime: Optional[dict]) -> Optional[str]:
    """Backwards-compatible string-only wrapper."""
    result = assess_regime_direction(side, regime)
    return result.reason if result.action == "BLOCK" else None


# ─────────────────────────────────────────────────────────────────────────────
# 7. PRE-EXECUTION MOMENTUM ALIGNMENT GUARD (M1 slope check at release time)
# ─────────────────────────────────────────────────────────────────────────────


def momentum_alignment_reason(
    side:   str,
    df_m1:  pd.DataFrame,
) -> Optional[str]:
    """
    Blocks execution when M1 momentum at release time is moving strongly
    against the signal direction. Catches the case where a signal was good
    when queued (M15 close), but reversed during the queue wait.

    Computes the average price change per M1 bar over the last N bars.
    Blocks BUY if price is falling at >= MIN_POINTS per bar.
    Blocks SELL if price is rising at >= MIN_POINTS per bar.

    Returns a reason string if blocked, or None if allowed.
    """
    if not MOMENTUM_ALIGNMENT_ENABLED:
        return None
    lookback = max(MOMENTUM_ALIGNMENT_LOOKBACK, 2)
    if df_m1 is None or len(df_m1) < lookback + 1:
        return None

    closes = df_m1["close"].astype(float).values
    avg_delta = float((closes[-1] - closes[-1 - lookback]) / lookback)

    side = str(side).upper()
    min_pts = MOMENTUM_ALIGNMENT_MIN_POINTS

    if side == "BUY" and avg_delta <= -min_pts:
        return (f"momentum_alignment BUY blocked: M1 avg delta "
                f"{avg_delta:+.2f} pts/bar ({lookback} bars) <= {min_pts} - "
                f"price falling, signal stale")
    if side == "SELL" and avg_delta >= min_pts:
        return (f"momentum_alignment SELL blocked: M1 avg delta "
                f"{avg_delta:+.2f} pts/bar ({lookback} bars) >= {min_pts} - "
                f"price rising, signal stale")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED HARD-BLOCK CHECK
# ─────────────────────────────────────────────────────────────────────────────

# Count of blocks suppressed by the trend-regime exemption (visibility for
# replays and live log audits).
trend_exemption_count = 0


def _trend_exemption_active(side: str, regime: Optional[dict]) -> bool:
    """
    True when the HMM regime confidently agrees with the signal direction:
    TREND_UP exempts BUY, TREND_DOWN exempts SELL — from the intraday
    extreme and H4 zone guards only. News blackout is never exempted.
    """
    if not GUARD_TREND_REGIME_EXEMPTION or not regime:
        return False
    rname = str(regime.get("regime", "")).upper()
    conf  = float(regime.get("confidence", 0.0))
    if conf < GUARD_EXEMPT_MIN_REGIME_CONF:
        return False
    side = str(side).upper()
    return (side == "BUY" and rname == "TREND_UP") or \
           (side == "SELL" and rname == "TREND_DOWN")


def hard_block_reason(
    side:          str,
    df_primary:    pd.DataFrame,
    df_h4:         Optional[pd.DataFrame],
    current_price: float,
    m15_atr:       float,
    now:           Optional[datetime] = None,
    regime:        Optional[dict] = None,
) -> Optional[str]:
    """
    Run all hard-block guards. Returns the first firing guard's reason,
    or None if the release is allowed.

    `regime` (dict with "regime" and "confidence", from the HMM router)
    enables the trend-regime exemption for the extreme/zone guards.
    """
    global trend_exemption_count

    reason = news_blackout_reason(now)
    if reason:
        return reason

    reason = session_open_block_reason(now)
    if reason:
        return reason

    exempt = _trend_exemption_active(side, regime)

    reason = intraday_extreme_reason(side, df_primary, current_price, m15_atr, now)
    if reason is None and df_h4 is not None:
        reason = h4_zone_reason(side, df_h4, current_price)

    if reason:
        if exempt:
            trend_exemption_count += 1
            logger.info(f"[GUARD] Trend-regime exemption ({regime.get('regime')} "
                        f"conf={regime.get('confidence', 0.0):.2f}) overrides: {reason}")
            return None
        return reason

    return None
