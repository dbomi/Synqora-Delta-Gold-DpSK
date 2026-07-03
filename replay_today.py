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
)
from data_engine import initialize_mt5, shutdown_mt5, fetch_latest, align_to_primary
from feature_engine import build_live_features
from model_stack import ModelStack
from regime_detector import RegimeRouter
from meta_agent import MetaAgent
from signal_queue import SignalQueue, QueuedSignal
from entry_guards import hard_block_reason

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


def main():
    enforce_guards = "--no-guards" not in sys.argv
    if not initialize_mt5():
        raise RuntimeError("MT5 connection failed. Open the terminal and log in.")

    try:
        # ── Load models ────────────────────────────────────────────────────
        stack  = ModelStack().load(MODELS_DIR)
        router = RegimeRouter().load(MODELS_DIR)
        with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)

        # ── Fetch data (enough M15 warmup for the 271-bar feature warmup) ──
        df15 = fetch_latest(SYMBOL, "M15", count=1200)
        dfh1 = fetch_latest(SYMBOL, "H1",  count=600)
        dfh4 = fetch_latest(SYMBOL, "H4",  count=300)
        dfm1 = fetch_latest(SYMBOL, "M1",  count=2000)
        dfm5 = fetch_latest(SYMBOL, "M5",  count=600)
        for name, d in [("M15", df15), ("H1", dfh1), ("H4", dfh4), ("M1", dfm1), ("M5", dfm5)]:
            if d.empty:
                raise RuntimeError(f"No {name} data from MT5.")

        today = datetime.now(timezone.utc).date()
        day_start = pd.Timestamp(today, tz="UTC")
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
                events_signals.append({
                    "time": t.strftime("%H:%M"), "buy_p": probs["buy_prob"],
                    "sell_p": probs["sell_prob"], "regime": regime["regime"],
                    "conf": regime["confidence"], "gate": "OPEN" if gate_ok else "CLOSED",
                    "action": decision.action if gate_ok else "NO_TRADE(regime)",
                })

                if gate_ok and decision.action in ("BUY", "SELL"):
                    # dedup: one enqueue per side per M15 bar window
                    le = last_enqueue[decision.action]
                    if le is None or (t - le) >= pd.Timedelta(minutes=15):
                        hl = d15["high"] - d15["low"]
                        hc = (d15["high"] - d15["close"].shift(1)).abs()
                        lc = (d15["low"]  - d15["close"].shift(1)).abs()
                        atr = float(pd.concat([hl, hc, lc], axis=1).max(axis=1)
                                    .ewm(span=14, adjust=False).mean().iloc[-1])
                        sig = QueuedSignal(
                            side=decision.action, family="GBM_M15",
                            source_cid=f"GBM-{decision.action}-{uuid.uuid4().hex[:6]}",
                            queue_price=price, queue_time=t.to_pydatetime(),
                            m15_atr=atr, queue_spread=spread,
                        )
                        queue.enqueue(sig)
                        last_enqueue[decision.action] = t

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
