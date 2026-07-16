"""
=============================================================================
REPLAY TODAY — SYNQORA DELTA GOLD FABLE
Dry-run of the full pipeline on TODAY's gold prices via the live MT5
connection. No orders are placed — execution is simulated.

For every M1 close of the current UTC day, chronologically:
  - On M15 closes: features -> GBM probs -> HMM regime gate -> meta gate
    -> enqueue signal (exactly like live).
  - Every M1 close: expire stale, score queued signals, apply hard blocks,
    release (score >= 4.0, <=3/cycle, <=2/side).
  - Released signals become simulated trades: SL=1.0xATR / TP=2.0xATR,
    resolved against subsequent M1 highs/lows (SL assumed first if both
    hit in one bar), negative time stop 24 M15 bars, max hold 48.

Usage:  python replay_today.py [--no-guards]
        --no-guards : hard blocks are logged as advisory but do NOT stop
                      releases (shows what the queue gate alone would do)
=============================================================================
"""

import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import (
    SYMBOL, PRIMARY_TF, CONTEXT_TFS, LOT_SIZE,
    MODELS_DIR, COOLDOWN_BARS, REGIME_MIN_CONFIDENCE,
    TRIPLE_BARRIER_TP_ATR, TRIPLE_BARRIER_SL_ATR,
    NEGATIVE_TIME_STOP_BARS, MAX_HOLD_BARS,
    REGIME_USE_TREND_DETECTOR,
    BREAKOUT_ENABLED, BREAKOUT_LOOKBACK_BARS,
    BREAKOUT_VOLUME_MULT, BREAKOUT_MIN_STRENGTH,
    DUAL_PATH_ENABLED, DUAL_PATH_ADX_MIN, DUAL_PATH_PROB_FALLBACK,
    DUAL_PATH_MAX_ATR_EXTENSION,
    DIRECTION_FILTER_ENABLED,
    DIRECTION_FILTER_ALIGN_BOOST, DIRECTION_FILTER_OPPOSE_PENALTY,
    DIRECTION_FILTER_LOT_ALIGN_BOOST, DIRECTION_FILTER_LOT_OPPOSE_PENALTY,
    DIRECTION_FILTER_MIN_CONFIDENCE,
    DIRECTION_FILTER_EMA_WEIGHT, DIRECTION_FILTER_MOMENTUM_WEIGHT,
    DIRECTION_FILTER_ACCEL_WEIGHT, DIRECTION_FILTER_DI_WEIGHT,
    DIRECTION_FILTER_SWING_WEIGHT,
    RATCHET_ENABLED, RATCHET_ARM_AT_R, RATCHET_LOCK_AT_R,
    RATCHET_VIRTUAL_LEVER,
)
from data_engine import initialize_mt5, shutdown_mt5, fetch_latest, align_to_primary
from feature_engine import build_live_features
from model_stack import ModelStack
from regime_detector import RegimeRouter, TrendRegimeDetector, detect_market_breakout, check_trend_structure, m30_direction_filter
from meta_agent import MetaAgent
from signal_queue import SignalQueue, QueuedSignal
from entry_guards import hard_block_reason, assess_regime_direction

import pickle
import os

logging.basicConfig(level=logging.WARNING)   # keep module logs quiet; we print our own report
log = print

USD_PER_PRICE_UNIT_PER_LOT = 100.0   # GOLD: 1.00 lot = 100 oz -> $100 per $1 move
POINT = 0.01


def closed_before(df: pd.DataFrame, t: pd.Timestamp, tf_minutes: int) -> pd.DataFrame:
    """Bars fully closed at time t (bar open time + duration <= t)."""
    return df[df.index <= t - pd.Timedelta(minutes=tf_minutes)]


def simulate_protection(tr: dict, dfm1: pd.DataFrame, arm_r: float, mode: str) -> dict:
    """
    What-if: replay one trade's M1 path with a protection rule.
      mode "BE"     : once peak favorable excursion >= arm_r * R, stop -> entry.
      mode "LOCK60" : once peak >= arm_r * R, stop ratchets to entry + 60% of peak.
    Stops update from a bar's extremes only AFTER exit checks on that bar
    (no intra-bar lookahead). TP/SL/time-stops otherwise as the baseline.
    """
    entry, sl, tp = tr["entry"], tr["sl"], tr["tp"]
    dirn = 1 if tr["side"] == "BUY" else -1
    r    = tr["r_price"]
    stop = sl
    peak = 0.0
    bars = dfm1[dfm1.index >= tr["entry_time"]]
    horizon_end = tr["entry_time"] + pd.Timedelta(minutes=MAX_HOLD_BARS * 15)
    exit_price, outcome = None, None

    for ts, bar in bars.iterrows():
        hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        adverse_touch = (lo <= stop) if dirn == 1 else (hi >= stop)
        if adverse_touch:
            exit_price = stop
            outcome = "SL" if abs(stop - sl) < 1e-9 else ("BE" if abs(stop - entry) < 1e-9 else "LOCK")
            break
        fav_touch = (hi >= tp) if dirn == 1 else (lo <= tp)
        if fav_touch:
            exit_price, outcome = tp, "TP"
            break

        held_min = (ts - tr["entry_time"]).total_seconds() / 60.0
        mtm = (close - entry) * dirn
        if held_min >= NEGATIVE_TIME_STOP_BARS * 15 and mtm < 0:
            exit_price, outcome = close, "NEG_TIME_STOP"
            break
        if ts >= horizon_end:
            exit_price, outcome = close, "MAX_HOLD"
            break

        # Update peak and ratchet the stop (takes effect next bar).
        fav = (hi - entry) * dirn
        if fav > peak:
            peak = fav
            if peak >= arm_r * r:
                if mode == "BE":
                    new_stop = entry
                else:  # LOCK60
                    new_stop = entry + dirn * 0.60 * peak
                if (new_stop - stop) * dirn > 0:
                    stop = new_stop

    if outcome is None:
        exit_price, outcome = float(bars["close"].iloc[-1]) if len(bars) else entry, "OPEN_EOD"

    pnl = ((exit_price - entry) * dirn - tr["spread_cost"]) * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT
    return {"outcome": outcome, "exit": exit_price, "pnl": pnl}


def main(day=None, fetch_mult=1):
    """day: date to replay (default today, UTC). fetch_mult scales history
    fetch depth so past days keep their full feature warmup."""
    enforce_guards = "--no-guards" not in sys.argv
    if not initialize_mt5():
        raise RuntimeError("MT5 connection failed. Open the terminal and log in.")

    try:
        # ── Load models ────────────────────────────────────────────────────
        stack  = ModelStack().load(MODELS_DIR)
        router_cls = TrendRegimeDetector if REGIME_USE_TREND_DETECTOR else RegimeRouter
        router = router_cls().load(MODELS_DIR)
        with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)

        # ── Fetch data (enough M15 warmup for the 271-bar feature warmup) ──
        df15 = fetch_latest(SYMBOL, "M15", count=1200 * fetch_mult)
        dfh1 = fetch_latest(SYMBOL, "H1",  count=600 * fetch_mult)
        dfh4 = fetch_latest(SYMBOL, "H4",  count=300 * fetch_mult)
        dfm1 = fetch_latest(SYMBOL, "M1",  count=2000 * fetch_mult)
        dfm5 = fetch_latest(SYMBOL, "M5",  count=600 * fetch_mult)
        dfm30 = fetch_latest(SYMBOL, "M30", count=300 * fetch_mult)
        for name, d in [("M15", df15), ("H1", dfh1), ("H4", dfh4), ("M1", dfm1), ("M5", dfm5)]:
            if d.empty:
                raise RuntimeError(f"No {name} data from MT5.")

        today = day or datetime.now(timezone.utc).date()
        day_start = pd.Timestamp(today, tz="UTC")
        day_end = day_start + pd.Timedelta(days=1)
        # trim ALL frames so the replay of a past day never sees later data
        df15, dfh1, dfh4 = (d[d.index < day_end] for d in (df15, dfh1, dfh4))
        dfm1, dfm5, dfm30 = (d[d.index < day_end] for d in (dfm1, dfm5, dfm30))
        m1_today = dfm1[dfm1.index >= day_start]
        if m1_today.empty:
            raise RuntimeError("No M1 bars for today yet.")

        log("=" * 78)
        log(f"SYNQORA DELTA GOLD FABLE — REPLAY OF {today} (UTC)")
        log(f"M1 bars today: {len(m1_today)}  ({m1_today.index[0]} -> {m1_today.index[-1]})")
        log(f"Price range today: {m1_today['low'].min():.2f} - {m1_today['high'].max():.2f}")
        log("=" * 78)

        # ── State ───────────────────────────────────────────────────────────
        queue      = SignalQueue()
        meta       = MetaAgent()
        bar_no     = 0
        sim_pnl    = 0.0
        open_trades  = []   # dicts: side, entry, sl, tp, entry_time, cid, score
        done_trades  = []
        events_signals  = []   # per-M15 signal decisions
        events_blocks   = []
        events_releases = []
        last_enqueue = {"BUY": None, "SELL": None}

        m15_closes = set(df15.index + pd.Timedelta(minutes=15))

        # ── Chronological replay over today's M1 closes ─────────────────────
        for t_open in m1_today.index:
            t = t_open + pd.Timedelta(minutes=1)      # M1 bar close time
            m1_closed = dfm1[dfm1.index <= t_open]    # this bar and earlier are closed at t
            last_m1   = m1_closed.iloc[-1]
            price     = float(last_m1["close"])
            spread    = float(last_m1.get("spread", 30.0))

            # ── Resolve open simulated trades on this M1 bar ────────────────
            still_open = []
            for tr in open_trades:
                hi, lo = float(last_m1["high"]), float(last_m1["low"])
                d = 1 if tr["side"] == "BUY" else -1
                fav = ((hi - tr["entry"]) if d == 1 else (tr["entry"] - lo))
                adv = ((tr["entry"] - lo) if d == 1 else (hi - tr["entry"]))
                if fav > tr["mfe"]:
                    tr["mfe"], tr["mfe_time"] = fav, t
                tr["mae"] = max(tr["mae"], adv)
                held_m15 = (t - tr["entry_time"]).total_seconds() / 60.0 / 15.0
                exit_price, outcome = None, None
                if tr["side"] == "BUY":
                    if lo <= tr["sl"]:            exit_price, outcome = tr["sl"], "SL"
                    elif hi >= tr["tp"]:          exit_price, outcome = tr["tp"], "TP"
                else:
                    if hi >= tr["sl"]:            exit_price, outcome = tr["sl"], "SL"
                    elif lo <= tr["tp"]:          exit_price, outcome = tr["tp"], "TP"
                if outcome is None and held_m15 >= NEGATIVE_TIME_STOP_BARS:
                    mtm = (price - tr["entry"]) * (1 if tr["side"] == "BUY" else -1)
                    if mtm < 0:                   exit_price, outcome = price, "NEG_TIME_STOP"
                if outcome is None and held_m15 >= MAX_HOLD_BARS:
                    exit_price, outcome = price, "MAX_HOLD"

                # ── P17: Two-stage trailing ratchet + virtual lever ────────
                if outcome is None and RATCHET_ENABLED:
                    r_price = tr.get("r_price", TRIPLE_BARRIER_SL_ATR * tr.get("atr", 0.0))
                    if r_price > 0:
                        peak_r = tr["mfe"] / r_price
                        if peak_r >= RATCHET_ARM_AT_R and not tr.get("r_locked", False):
                            lock_sl = tr["entry"] + d * RATCHET_LOCK_AT_R * r_price
                            if (d == 1 and lock_sl > tr["sl"]) or (d == -1 and lock_sl < tr["sl"]):
                                tr["sl"] = lock_sl
                                tr["r_locked"] = True
                            if RATCHET_VIRTUAL_LEVER:
                                if (d == 1 and lo <= lock_sl) or (d == -1 and hi >= lock_sl):
                                    exit_price, outcome = lock_sl, "VLEV"

                if outcome:
                    direction = 1 if tr["side"] == "BUY" else -1
                    move = (exit_price - tr["entry"]) * direction - tr["spread_cost"]
                    pnl  = move * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT
                    sim_pnl += pnl
                    tr.update(exit_time=t, exit_price=exit_price, outcome=outcome, pnl=pnl)
                    done_trades.append(tr)
                else:
                    still_open.append(tr)
            open_trades = still_open

            # ── M15 close → signal generation ───────────────────────────────
            if t in m15_closes and t >= day_start:
                bar_no += 1
                d15 = closed_before(df15, t, 15)
                dh1 = closed_before(dfh1, t, 60)
                dh4 = closed_before(dfh4, t, 240)
                if len(d15) < 400:
                    continue
                data_aligned = align_to_primary({"M15": d15, "H1": dh1, "H4": dh4}, "M15")
                feats = build_live_features(data_aligned, feature_cols, "M15")
                if feats is None or feats.empty:
                    continue

                regime = router.get_regime(d15)
                probs  = stack.predict(feats)

                open_pos_df = pd.DataFrame([{"type": x["side"]} for x in open_trades]) \
                              if open_trades else pd.DataFrame()
                decision = meta.decide(probs, regime, open_pos_df, sim_pnl,
                                       current_bar=bar_no, cooldown_bars=COOLDOWN_BARS)

                gate_ok = regime["trade_ok"] and regime["confidence"] >= REGIME_MIN_CONFIDENCE
                # Option 2: Market-structure breakout override when gate is closed
                breakout_override = None
                if not gate_ok and BREAKOUT_ENABLED:
                    try:
                        breakout_override = detect_market_breakout(
                            lookback_bars=BREAKOUT_LOOKBACK_BARS,
                            vol_mult=BREAKOUT_VOLUME_MULT,
                            min_strength=BREAKOUT_MIN_STRENGTH,
                        )
                    except Exception as e:
                        log(f"[BREAKOUT] detection failed: {e}")
                    if breakout_override is not None:
                        gate_ok = True
                        log(f"[BREAKOUT] {breakout_override['direction']} breakout "
                            f"str={breakout_override['strength']:.2f} "
                            f"— overriding regime gate")

                events_signals.append({
                    "time": t.strftime("%H:%M"), "buy_p": probs["buy_prob"],
                    "sell_p": probs["sell_prob"], "regime": regime["regime"],
                    "conf": regime["confidence"], "gate": "OPEN" if gate_ok else "CLOSED",
                    "action": decision.action if gate_ok else "NO_TRADE(regime)",
                })

                if gate_ok:
                    # Determine signal action: from meta-agent or Option 3 dual-path
                    signal_action = decision.action if decision.action in ("BUY", "SELL") else None
                    if signal_action is None and DUAL_PATH_ENABLED:
                        try:
                            signal_action = check_trend_structure(
                                d15, adx_min=DUAL_PATH_ADX_MIN,
                                max_atr_extension=DUAL_PATH_MAX_ATR_EXTENSION)
                        except Exception as e:
                            log(f"[DUAL_PATH] check failed: {e}")
                        if signal_action is not None:
                            log(f"[DUAL_PATH] {signal_action} ADX+EMA signal "
                                f"(prob={DUAL_PATH_PROB_FALLBACK:.2f})")

                    # ── Option 4: M30 Trend-Context Modulator ───────────────
                    m30_prob_factor = 1.0
                    m30_lot_mult = 1.0
                    if signal_action is not None and DIRECTION_FILTER_ENABLED:
                        try:
                            dm30 = closed_before(dfm30, t, 30)
                            m30_dir = m30_direction_filter(
                                dm30.tail(300) if not dm30.empty else pd.DataFrame(),
                                min_swing_bars=5,
                                ema_weight=DIRECTION_FILTER_EMA_WEIGHT,
                                momentum_weight=DIRECTION_FILTER_MOMENTUM_WEIGHT,
                                accel_weight=DIRECTION_FILTER_ACCEL_WEIGHT,
                                di_weight=DIRECTION_FILTER_DI_WEIGHT,
                                swing_weight=DIRECTION_FILTER_SWING_WEIGHT,
                            )
                        except Exception as e:
                            m30_dir = {"direction": "NEUTRAL", "confidence": 0.0}
                            log(f"[DIR_FILTER] check failed: {e}")
                        m30_conf = m30_dir.get("confidence", 0.0)
                        if m30_conf >= DIRECTION_FILTER_MIN_CONFIDENCE:
                            if m30_dir["direction"] == signal_action:
                                m30_prob_factor = 1.0 + m30_conf * DIRECTION_FILTER_ALIGN_BOOST
                                m30_lot_mult = 1.0 + m30_conf * DIRECTION_FILTER_LOT_ALIGN_BOOST
                                log(f"[DIR_FILTER] ALIGNED {signal_action} "
                                    f"(m30_conf={m30_conf:.2f}, prob_f={m30_prob_factor:.3f}, lot_m={m30_lot_mult:.3f})")
                            elif m30_dir["direction"] != "NEUTRAL":
                                m30_prob_factor = max(0.1, 1.0 - m30_conf * DIRECTION_FILTER_OPPOSE_PENALTY)
                                m30_lot_mult = max(0.1, 1.0 - m30_conf * DIRECTION_FILTER_LOT_OPPOSE_PENALTY)
                                log(f"[DIR_FILTER] OPPOSED {signal_action} vs M30 {m30_dir['direction']} "
                                    f"(m30_conf={m30_conf:.2f}, prob_f={m30_prob_factor:.3f}, lot_m={m30_lot_mult:.3f})")

                    if signal_action is not None:
                        is_dual_path = decision.action not in ("BUY", "SELL")
                        family = "TREND_M15" if is_dual_path else "GBM_M15"
                        buy_prob = DUAL_PATH_PROB_FALLBACK if (is_dual_path and signal_action == "BUY") else float(probs["buy_prob"])
                        sell_prob = DUAL_PATH_PROB_FALLBACK if (is_dual_path and signal_action == "SELL") else float(probs["sell_prob"])
                        # Apply M30 probability modulation (affects A+ dual-entry)
                        if m30_prob_factor != 1.0:
                            buy_prob  = min(1.0, buy_prob * m30_prob_factor)
                            sell_prob = min(1.0, sell_prob * m30_prob_factor)

                        # dedup: one enqueue per side per M15 bar window
                        le = last_enqueue[signal_action]
                        if le is None or (t - le) >= pd.Timedelta(minutes=15):
                            hl = d15["high"] - d15["low"]
                            hc = (d15["high"] - d15["close"].shift(1)).abs()
                            lc = (d15["low"]  - d15["close"].shift(1)).abs()
                            atr = float(pd.concat([hl, hc, lc], axis=1).max(axis=1)
                                        .ewm(span=14, adjust=False).mean().iloc[-1])
                            qprice = price
                            sig = QueuedSignal(
                                side=signal_action, family=family,
                                source_cid=f"{family}-{signal_action}-{uuid.uuid4().hex[:6]}",
                                queue_price=qprice, queue_time=t.to_pydatetime(),
                                m15_atr=atr, queue_spread=spread,
                                meta={"buy_prob": buy_prob, "sell_prob": sell_prob,
                                      "m30_lot_mult": m30_lot_mult,
                                      "regime": regime["regime"],
                                      "regime_conf": regime["confidence"]},
                            )
                            queue.enqueue(sig)
                            last_enqueue[signal_action] = t

            # ── M1 release cycle ─────────────────────────────────────────────
            if len(queue) > 0:
                d15_g = closed_before(df15, t, 15)
                dh4_g = closed_before(dfh4, t, 240)
                dm5_g = closed_before(dfm5, t, 5)

                def hard_block(sig, _t=t, _d15=d15_g, _dh4=dh4_g, _p=price):
                    r = hard_block_reason(sig.side, _d15, _dh4, _p, sig.m15_atr,
                                          now=_t.to_pydatetime())
                    if r:
                        events_blocks.append({"time": _t.strftime("%H:%M"),
                                              "side": sig.side, "reason": r})
                    return r if enforce_guards else None

                released = queue.release_cycle(
                    df_m1=m1_closed.tail(80), df_m5=dm5_g.tail(20),
                    current_price=price, current_spread=spread,
                    hard_block_fn=hard_block, now=t.to_pydatetime(),
                )
                for item in released:
                    sig = item["signal"]

                    # Regime direction gate at release time (matches live_trader)
                    _prob_key = "buy_prob" if sig.side == "BUY" else "sell_prob"
                    _other_key = "sell_prob" if sig.side == "BUY" else "buy_prob"
                    _prob = float(sig.meta.get(_prob_key, 0.0))
                    _edge = _prob - float(sig.meta.get(_other_key, 0.0))
                    _score = float(item.get("score", 0.0))
                    _rd = assess_regime_direction(
                        sig.side, regime,
                        df_m15=d15_g.tail(200), df_m5=dm5_g, df_m1=m1_closed.tail(80),
                        prob=_prob, edge=_edge, queue_score=_score,
                    )
                    if _rd.action == "BLOCK":
                        continue

                    direction = 1 if sig.side == "BUY" else -1
                    entry = price
                    sl = entry - direction * TRIPLE_BARRIER_SL_ATR * sig.m15_atr
                    tp = entry + direction * TRIPLE_BARRIER_TP_ATR * sig.m15_atr
                    open_trades.append({
                        "side": sig.side, "cid": sig.source_cid, "entry": entry,
                        "sl": sl, "tp": tp, "entry_time": t, "score": item["score"],
                        "components": item["components"],
                        "spread_cost": spread * POINT,
                        "queue_price": sig.queue_price,
                        "queue_time": pd.Timestamp(sig.queue_time),
                        "r_price": TRIPLE_BARRIER_SL_ATR * sig.m15_atr,
                        "mfe": 0.0, "mae": 0.0, "mfe_time": None,
                    })
                    meta.record_executed_signal(sig.side, bar_no)
                    events_releases.append({
                        "time": t.strftime("%H:%M"), "side": sig.side,
                        "cid": sig.source_cid, "score": item["score"],
                        "entry": entry, "sl": sl, "tp": tp,
                        "components": item["components"],
                    })

        # ══ REPORT ═══════════════════════════════════════════════════════════
        log("\n--- M15 SIGNAL DECISIONS (today) " + "-" * 44)
        log(f"{'time':>5} {'buy_p':>6} {'sell_p':>7} {'regime':>10} {'conf':>5} "
            f"{'gate':>6}  action")
        for e in events_signals:
            log(f"{e['time']:>5} {e['buy_p']:>6.3f} {e['sell_p']:>7.3f} "
                f"{e['regime']:>10} {e['conf']:>5.2f} {e['gate']:>6}  {e['action']}")

        n_sig = sum(1 for e in events_signals if e["action"] in ("BUY", "SELL"))
        log(f"\nM15 bars evaluated: {len(events_signals)} | signals enqueued: {n_sig}")

        log("\n--- HARD BLOCKS DURING RELEASE " + "-" * 46)
        if events_blocks:
            seen = set()
            for b in events_blocks:
                key = (b["side"], b["reason"].split(" ")[0])
                if key in seen:
                    continue
                seen.add(key)
                log(f"  {b['time']} {b['side']}: {b['reason']}")
            log(f"  ({len(events_blocks)} block events total, deduplicated above)")
        else:
            log("  none")

        log("\n--- RELEASES (queue gate passed) " + "-" * 44)
        if events_releases:
            for r in events_releases:
                comp = ", ".join(f"{k}+{v:.1f}" for k, v in r["components"].items())
                log(f"  {r['time']} {r['side']} {r['cid']} score={r['score']:.1f} "
                    f"entry={r['entry']:.2f} SL={r['sl']:.2f} TP={r['tp']:.2f}")
                log(f"         [{comp}]")
        else:
            log("  none")

        log("\n--- SIMULATED TRADE RESULTS " + "-" * 49)
        if done_trades or open_trades:
            for tr in done_trades:
                log(f"  {tr['entry_time'].strftime('%H:%M')} {tr['side']} "
                    f"entry={tr['entry']:.2f} -> {tr['outcome']} @ {tr['exit_price']:.2f} "
                    f"({tr['exit_time'].strftime('%H:%M')})  pnl={tr['pnl']:+.2f} USD")
            for tr in open_trades:
                last_price = float(m1_today['close'].iloc[-1])
                direction = 1 if tr["side"] == "BUY" else -1
                mtm = ((last_price - tr["entry"]) * direction - tr["spread_cost"]) \
                      * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT
                log(f"  {tr['entry_time'].strftime('%H:%M')} {tr['side']} "
                    f"entry={tr['entry']:.2f} -> STILL OPEN  mtm={mtm:+.2f} USD")
            closed_pnl = sum(t_["pnl"] for t_ in done_trades)
            log(f"\n  Closed trades: {len(done_trades)} | open: {len(open_trades)} "
                f"| closed P&L @ {LOT_SIZE} lot: {closed_pnl:+.2f} USD")
        else:
            log("  no trades released today")

        # ── MFE / profit-protection analysis ────────────────────────────────
        if done_trades:
            log("\n--- MFE ANALYSIS (how green did each trade get before exit?) " + "-" * 15)
            log(f"{'entry':>6} {'side':>4} {'outcome':>14} {'MFE$':>7} {'MFE(R)':>7} "
                f"{'@peak':>6} {'MAE(R)':>7} {'final$':>8}")
            for tr in done_trades:
                mfe_usd = tr["mfe"] * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT
                mfe_r   = tr["mfe"] / tr["r_price"]
                mae_r   = tr["mae"] / tr["r_price"]
                pk      = tr["mfe_time"].strftime("%H:%M") if tr["mfe_time"] is not None else "-"
                log(f"{tr['entry_time'].strftime('%H:%M'):>6} {tr['side']:>4} "
                    f"{tr['outcome']:>14} {mfe_usd:>7.2f} {mfe_r:>7.2f} {pk:>6} "
                    f"{mae_r:>7.2f} {tr['pnl']:>8.2f}")

            losers = [t_ for t_ in done_trades if t_["outcome"] in ("SL", "NEG_TIME_STOP")]
            green_any  = [t_ for t_ in losers if t_["mfe"] > t_["spread_cost"]]
            green_03   = [t_ for t_ in losers if t_["mfe"] >= 0.3 * t_["r_price"]]
            green_05   = [t_ for t_ in losers if t_["mfe"] >= 0.5 * t_["r_price"]]
            green_10   = [t_ for t_ in losers if t_["mfe"] >= 1.0 * t_["r_price"]]
            log(f"\n  Losing trades: {len(losers)} | were green (net of spread): "
                f"{len(green_any)} | reached +0.3R: {len(green_03)} | +0.5R: "
                f"{len(green_05)} | +1.0R: {len(green_10)}")

            log("\n--- DELAY GATE CONTRIBUTION (queue price -> release entry) " + "-" * 17)
            improves = []
            for tr in done_trades:
                d = 1 if tr["side"] == "BUY" else -1
                impr = (tr["queue_price"] - tr["entry"]) * d
                lag  = (tr["entry_time"] - tr["queue_time"]).total_seconds() / 60.0
                improves.append(impr)
                log(f"  {tr['entry_time'].strftime('%H:%M')} {tr['side']}: queued "
                    f"{tr['queue_time'].strftime('%H:%M')} @ {tr['queue_price']:.2f} -> "
                    f"entry {tr['entry']:.2f} after {lag:.0f} min | entry "
                    f"{'improved' if impr > 0 else 'worse'} by {abs(impr):.2f} "
                    f"({impr / tr['r_price']:+.2f}R)")
            avg_impr = float(np.mean(improves))
            log(f"  Average entry improvement: {avg_impr:+.2f} "
                f"({avg_impr * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT:+.2f} USD/trade @ {LOT_SIZE} lot)")

            log("\n--- PROFIT-PROTECTION WHAT-IF (same entries, M1 resolution) " + "-" * 16)
            variants = [("BE@0.5R", 0.5, "BE"), ("BE@1.0R", 1.0, "BE"),
                        ("LOCK60@1R", 1.0, "LOCK60")]
            log(f"{'entry':>6} {'side':>4} {'RAW':>10}" +
                "".join(f" {name:>16}" for name, _, _ in variants))
            totals = {name: 0.0 for name, _, _ in variants}
            for tr in done_trades:
                row = f"{tr['entry_time'].strftime('%H:%M'):>6} {tr['side']:>4} " \
                      f"{tr['pnl']:>+7.2f} {tr['outcome'][:3]:>2}"
                for name, arm, mode in variants:
                    v = simulate_protection(tr, dfm1, arm, mode)
                    totals[name] += v["pnl"]
                    row += f" {v['pnl']:>+9.2f} {v['outcome'][:4]:>6}"
                log(row)
            raw_total = sum(t_["pnl"] for t_ in done_trades)
            log(f"\n  TOTALS: RAW {raw_total:+.2f}" +
                "".join(f" | {name} {totals[name]:+.2f}" for name, _, _ in variants))

            ledger = pd.DataFrame([{
                "queue_time": t_["queue_time"], "entry_time": t_["entry_time"],
                "side": t_["side"], "cid": t_["cid"], "score": t_["score"],
                "queue_price": t_["queue_price"], "entry": t_["entry"],
                "sl": t_["sl"], "tp": t_["tp"], "outcome": t_["outcome"],
                "exit_price": t_.get("exit_price"), "exit_time": t_.get("exit_time"),
                "mfe": t_["mfe"], "mae": t_["mae"], "r_price": t_["r_price"],
                "pnl_usd": t_["pnl"],
            } for t_ in done_trades])
            ledger_path = os.path.join("logs", f"replay_trades_{today}.csv")
            ledger.to_csv(ledger_path, index=False)
            log(f"\n  Trade ledger saved: {ledger_path}")

        log("\n--- QUEUE STATE AT END OF REPLAY " + "-" * 44)
        snap = queue.snapshot()
        if snap:
            for s in snap:
                log(f"  {s['side']} {s['cid']} queued {s['queue_time'][11:16]} "
                    f"@ {s['queue_price']:.2f} (age {s['age_min']:.0f} min)")
        else:
            log("  empty")
        log("=" * 78)

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
