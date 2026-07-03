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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import (
    BASE_DIR,
    NEWS_BLACKOUT_ENABLED, NEWS_STATIC_WINDOWS_UTC, NEWS_EVENTS_FILE,
    NEWS_PRE_EVENT_MINUTES, NEWS_POST_EVENT_MINUTES,
    INTRADAY_EXTREME_GUARD_ENABLED, INTRADAY_EXTREME_ATR_MULT,
    H4_TOPZONE_GUARD_ENABLED, H4_TOPZONE_LOOKBACK_BARS, H4_TOPZONE_UPPER_PCT,
    H4_BOTTOMZONE_GUARD_ENABLED, H4_BOTTOMZONE_LOWER_PCT,
    GUARD_TREND_REGIME_EXEMPTION, GUARD_EXEMPT_MIN_REGIME_CONF,
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
