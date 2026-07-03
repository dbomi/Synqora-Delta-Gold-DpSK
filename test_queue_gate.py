"""
Smoke test for the Fable Signal Queue Gate + hard-block guards.
Runs without MT5 — synthetic M1/M5/M15/H4 data only.

    python test_queue_gate.py
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from signal_queue import SignalQueue, QueuedSignal, score_signal
from entry_guards import intraday_extreme_reason, h4_zone_reason, news_blackout_reason

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

NOW = datetime.now(timezone.utc)
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, PASS if cond else FAIL))
    print(f"  [{PASS if cond else FAIL}] {name}")


def make_m1_bullish_confirmation(n=40, base=3300.0):
    """M1 bars ending with a strong bullish confirmation bar after a dip."""
    idx = pd.date_range(end=NOW, periods=n, freq="1min", tz="UTC")
    close = np.full(n, base) + np.cumsum(np.random.default_rng(7).normal(0, 0.05, n))
    # Engineer a dip then a turn: last 6 bars fall, final bar rips up.
    close[-7:-1] = close[-8] - np.linspace(0.5, 2.5, 6)
    close[-1] = close[-2] + 3.0                      # strong up close → ROC-3 flips positive
    open_ = np.roll(close, 1); open_[0] = close[0]
    high  = np.maximum(open_, close) + 0.10
    low   = np.minimum(open_, close) - 0.10
    vol   = np.full(n, 100.0); vol[-1] = 400.0       # 4× spike
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "tick_volume": vol}, index=idx)


def make_m5_bullish(n=10, base=3300.0):
    idx = pd.date_range(end=NOW, periods=n, freq="5min", tz="UTC")
    open_ = np.full(n, base); close = open_ + 1.0
    return pd.DataFrame({"open": open_, "high": close + 0.2, "low": open_ - 0.2,
                         "close": close, "tick_volume": np.full(n, 500.0)}, index=idx)


print("── 1. FIFO queue mechanics ──")
q = SignalQueue(capacity=20)
for i in range(22):
    q.enqueue(QueuedSignal(side="BUY" if i % 2 == 0 else "SELL", family="GBM_M15",
                           source_cid=f"cid-{i}", queue_price=3300.0,
                           queue_time=NOW, m15_atr=5.0))
check("capacity capped at 20", len(q) == 20)
check("oldest evicted first (cid-0, cid-1 gone)",
      all(s["cid"] not in ("cid-0", "cid-1") for s in q.snapshot()))

print("── 2. Expiry ──")
q2 = SignalQueue()
q2.enqueue(QueuedSignal(side="BUY", family="GBM_M15", source_cid="old",
                        queue_price=3300.0, queue_time=NOW - timedelta(minutes=120), m15_atr=5.0))
q2.enqueue(QueuedSignal(side="BUY", family="GBM_M15", source_cid="fresh",
                        queue_price=3300.0, queue_time=NOW, m15_atr=5.0))
expired = q2.expire_stale()
check("stale signal expired, fresh kept",
      len(expired) == 1 and expired[0].source_cid == "old" and len(q2) == 1)

print("── 3. Leading-indicator scoring (engineered BUY confirmation) ──")
df_m1 = make_m1_bullish_confirmation()
df_m5 = make_m5_bullish()
sig = QueuedSignal(side="BUY", family="GBM_M15", source_cid="score-test",
                   queue_price=float(df_m1["close"].iloc[-1]) + 5.0,  # price pulled back below queue price
                   queue_time=NOW, m15_atr=5.0, queue_spread=30.0)
res = score_signal(sig, df_m1, df_m5, current_price=float(df_m1["close"].iloc[-1]),
                   current_spread=20.0)  # 20 <= 0.8×30 → spread tightening
print(f"  score={res['score']:.1f} components={res['components']}")
check("zero-cross detected", "m1_momentum_zero_cross" in res["components"])
check("volume spike detected", "tick_volume_spike" in res["components"])
check("body proportion detected", "candle_body_proportion" in res["components"])
check("pullback detected", "pullback_to_queue_price" in res["components"])
check("M5 alignment detected", "m5_alignment" in res["components"])
check("spread tightening detected", "spread_tightening" in res["components"])
check("score above release threshold 4.0", res["score"] >= 4.0)

sell_sig = QueuedSignal(side="SELL", family="GBM_M15", source_cid="sell-test",
                        queue_price=3200.0, queue_time=NOW, m15_atr=5.0, queue_spread=30.0)
res_sell = score_signal(sell_sig, df_m1, df_m5,
                        current_price=float(df_m1["close"].iloc[-1]), current_spread=30.0)
print(f"  SELL score on bullish tape = {res_sell['score']:.1f}")
check("SELL not confirmed on bullish tape", res_sell["score"] < 4.0)

print("── 4. Release caps (3 total, 2 per side, best-first) ──")
q3 = SignalQueue()
for i in range(4):
    q3.enqueue(QueuedSignal(side="BUY", family="GBM_M15", source_cid=f"buy-{i}",
                            queue_price=9999.0, queue_time=NOW, m15_atr=5.0, queue_spread=30.0))
released = q3.release_cycle(df_m1, df_m5,
                            current_price=float(df_m1["close"].iloc[-1]),
                            current_spread=20.0, hard_block_fn=None)
check("max 2 BUY released per cycle", len(released) == 2
      and all(r["signal"].side == "BUY" for r in released))
check("unreleased signals stay queued", len(q3) == 2)

print("── 5. Hard-block guards ──")
idx15 = pd.date_range(start=pd.Timestamp(NOW.date(), tz="UTC"), periods=30, freq="15min")
df15 = pd.DataFrame({"open": 3300.0, "high": 3310.0, "low": 3290.0, "close": 3305.0},
                    index=idx15)
# BUY at 3308 with ATR 5 → within 1.5×5=7.5 of session high 3310 → blocked
r = intraday_extreme_reason("BUY", df15, current_price=3308.0, m15_atr=5.0)
check("intraday extreme blocks BUY near session high", r is not None)
r = intraday_extreme_reason("BUY", df15, current_price=3295.0, m15_atr=1.0)
check("intraday extreme allows BUY away from high", r is None)
r = intraday_extreme_reason("SELL", df15, current_price=3291.0, m15_atr=5.0)
check("intraday extreme blocks SELL near session low", r is not None)

idx4 = pd.date_range(end=NOW, periods=25, freq="4h", tz="UTC")
df_h4 = pd.DataFrame({"open": 3300.0, "high": 3350.0, "low": 3250.0, "close": 3300.0},
                     index=idx4)
check("H4 topzone blocks BUY at top of range",
      h4_zone_reason("BUY", df_h4, current_price=3345.0) is not None)
check("H4 topzone allows BUY mid-range",
      h4_zone_reason("BUY", df_h4, current_price=3300.0) is None)
check("H4 bottomzone blocks SELL at bottom of range",
      h4_zone_reason("SELL", df_h4, current_price=3255.0) is not None)

import entry_guards
entry_guards.NEWS_STATIC_WINDOWS_UTC = [(f"{NOW.hour:02d}:00", f"{(NOW.hour + 1) % 24:02d}:00")]
check("news static window blocks now", news_blackout_reason(NOW) is not None)
entry_guards.NEWS_STATIC_WINDOWS_UTC = []
check("no news window → allowed", news_blackout_reason(NOW) is None)

print("── 6. Hard block prevents release regardless of score ──")
q4 = SignalQueue()
q4.enqueue(QueuedSignal(side="BUY", family="GBM_M15", source_cid="blocked",
                        queue_price=9999.0, queue_time=NOW, m15_atr=5.0, queue_spread=30.0))
released = q4.release_cycle(df_m1, df_m5,
                            current_price=float(df_m1["close"].iloc[-1]),
                            current_spread=20.0,
                            hard_block_fn=lambda s: "news_blackout_test")
check("high-score signal not released under hard block", len(released) == 0)
check("hard-blocked signal remains queued for later cycles", len(q4) == 1)

print()
n_fail = sum(1 for _, r in results if r == FAIL)
print(f"{'='*50}\n{len(results) - n_fail}/{len(results)} checks passed"
      + ("" if n_fail == 0 else f" — {n_fail} FAILED"))
raise SystemExit(1 if n_fail else 0)
