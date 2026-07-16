"""
=============================================================================
REPLAY RANGE — SYNQORA DELTA GOLD FABLE
Multi-day dry-run of the full pipeline on recent gold history via MT5.
No orders are placed.

Same event flow as live/replay_today (M15 signal generation -> queue ->
M1 leading-indicator release -> simulated SL/TP fills), plus aggregate
analytics across the sample:
  - MFE/MAE per trade: how green did losers get before dying?
  - Profit-protection what-ifs: BE@0.5R, BE@1.0R, LOCK60@1R
  - Chase analysis: entry vs queue price in ATR (max-chase what-if)
  - Per-day and total P&L, with/without hard-block guards

Speed note: features and GBM probabilities are precomputed in one batch
over the whole range (same causal convention used in training/validation);
regime, queue scoring, guards, and fills are replayed bar-by-bar.

Usage:  python replay_range.py [--days N] [--no-guards]
                               [--start YYYY-MM-DD --end YYYY-MM-DD]
                               [--out ledger.csv]
        --days N       trading days to replay counting back from now
                       (default 15; ignored when --start/--end given)
        --start/--end  replay an explicit historical window (fetched by
                       date range, subject to broker history depth)
        --no-guards    hard blocks logged as advisory only
        --out          explicit ledger CSV path
=============================================================================
"""

import logging
import os
import pickle
import sys
import uuid
from datetime import timezone

import numpy as np
import pandas as pd

from config import (
    SYMBOL, LOT_SIZE, MODELS_DIR, COOLDOWN_BARS, REGIME_MIN_CONFIDENCE,
    TRIPLE_BARRIER_TP_ATR, TRIPLE_BARRIER_SL_ATR,
    NEGATIVE_TIME_STOP_BARS, MAX_HOLD_BARS,
    DAILY_PROFIT_TARGET, DAILY_LOSS_LIMIT,
    USE_FRESH_ATR_GEOMETRY, ENSEMBLE_COHERENCE_CHECK,
    CONSECUTIVE_LOSS_REDUCTION, DIRECTIONAL_LOSS_LIMIT_ENABLED,
    MFE_TRAIL_ENABLED, MFE_ARM_BE_AT_R, MFE_TRAIL_ACTIVATE_AT_R,
    MFE_TRAIL_DISTANCE_R, MFE_TRAIL_BUFFER_POINTS,
    MFE_TRAIL_AUTO_MODE, MFE_TRAIL_ATR_RATIO, MFE_TRAIL_ATR_LOOKBACK,
    MFE_TRAIL_REGIME_TRIGGERS,
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
from data_engine import (initialize_mt5, shutdown_mt5, fetch_latest,
                         fetch_ohlcv, align_to_primary)
from feature_engine import build_live_features
from model_stack import ModelStack
from regime_detector import RegimeRouter, TrendRegimeDetector, detect_market_breakout, check_trend_structure, m30_direction_filter
from meta_agent import MetaAgent
from signal_queue import SignalQueue, QueuedSignal
import entry_guards
from entry_guards import hard_block_reason

logging.basicConfig(level=logging.ERROR)
log = print

USD_PER_PRICE_UNIT_PER_LOT = 100.0
POINT = 0.01


def ewm_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=period, adjust=False).mean()


def simulate_protection(tr: dict, dfm1: pd.DataFrame, arm_r: float, mode: str) -> dict:
    """What-if replay of one trade with a protection rule (see replay_today).

    Modes:
      BE     – move SL to entry at MFE >= arm_r × R
      LOCK60 – move SL to lock 60% of peak when MFE >= arm_r × R
      TRAIL  – (P11) arm BE at MFE >= MFE_ARM_BE_AT_R, then trail
               stop MFE_TRAIL_DISTANCE_R behind max peak beyond
               MFE_TRAIL_ACTIVATE_AT_R.
      P12    – close at market when profit (in $) >= arm_r (static target).
    """
    entry, sl, tp = tr["entry"], tr["sl"], tr["tp"]
    dirn = 1 if tr["side"] == "BUY" else -1
    r = tr["r_price"]
    stop, peak = sl, 0.0
    be_armed = False
    trailing = False
    bars = dfm1[dfm1.index >= tr["entry_time"]]
    horizon_end = tr["entry_time"] + pd.Timedelta(minutes=MAX_HOLD_BARS * 15)
    exit_price, outcome = None, None

    for ts, bar in bars.iterrows():
        hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        if (lo <= stop) if dirn == 1 else (hi >= stop):
            exit_price = stop
            outcome = "SL" if abs(stop - sl) < 1e-9 else ("BE" if abs(stop - entry) < 1e-9 else "LOCK")
            break
        if (hi >= tp) if dirn == 1 else (lo <= tp):
            exit_price, outcome = tp, "TP"
            break
        held_min = (ts - tr["entry_time"]).total_seconds() / 60.0
        if held_min >= NEGATIVE_TIME_STOP_BARS * 15 and (close - entry) * dirn < 0:
            exit_price, outcome = close, "NEG"
            break
        if ts >= horizon_end:
            exit_price, outcome = close, "HOLD"
            break
        fav = (hi - entry) * dirn
        if fav > peak:
            peak = fav
            if mode == "TRAIL" and MFE_TRAIL_ENABLED:
                if not be_armed and peak >= MFE_ARM_BE_AT_R * r:
                    be_sl = entry
                    if (be_sl - stop) * dirn > 0:
                        stop = be_sl
                        be_armed = True
                if peak >= MFE_TRAIL_ACTIVATE_AT_R * r:
                    trailing = True
                    trail_stop = entry + dirn * (peak - MFE_TRAIL_DISTANCE_R * r)
                    if (trail_stop - stop) * dirn > 0:
                        stop = trail_stop
            elif mode in ("BE", "LOCK60") and peak >= arm_r * r:
                new_stop = entry if mode == "BE" else entry + dirn * 0.60 * peak
                if (new_stop - stop) * dirn > 0:
                    stop = new_stop

        # P12: static target close — close at market when profit >= arm_r ($)
        if mode == "P12":
            profit = (close - entry) * dirn * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT
            if profit >= arm_r:
                exit_price, outcome = close, f"P12${int(arm_r)}"
                break

        if outcome is None:
            exit_price, outcome = (float(bars["close"].iloc[-1]) if len(bars) else entry), "END"
    pnl = ((exit_price - entry) * dirn - tr["spread_cost"]) * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT
    return {"outcome": outcome, "exit": exit_price, "pnl": pnl}


def main():
    days = 15
    if "--days" in sys.argv:
        days = int(sys.argv[sys.argv.index("--days") + 1])
    days = max(2, min(days, 25))
    enforce_guards = "--no-guards" not in sys.argv
    start_s = sys.argv[sys.argv.index("--start") + 1] if "--start" in sys.argv else None
    end_s   = sys.argv[sys.argv.index("--end") + 1] if "--end" in sys.argv else None
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    range_mode = start_s is not None and end_s is not None

    if not initialize_mt5():
        raise RuntimeError("MT5 connection failed. Open the terminal and log in.")

    try:
        stack = ModelStack().load(MODELS_DIR)
        router_cls = TrendRegimeDetector if REGIME_USE_TREND_DETECTOR else RegimeRouter
        router = router_cls().load(MODELS_DIR)
        with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)

        # ── Fetch ────────────────────────────────────────────────────────────
        if range_mode:
            from datetime import datetime as _dt, timedelta as _td
            start_dt = _dt.strptime(start_s, "%Y-%m-%d")
            end_dt   = _dt.strptime(end_s, "%Y-%m-%d") + _td(days=1)
            df15 = fetch_ohlcv(SYMBOL, "M15", start_dt - _td(days=45), end_dt)
            dfh1 = fetch_ohlcv(SYMBOL, "H1",  start_dt - _td(days=45), end_dt)
            dfh4 = fetch_ohlcv(SYMBOL, "H4",  start_dt - _td(days=90), end_dt)
            dfm1 = fetch_ohlcv(SYMBOL, "M1",  start_dt - _td(days=4),  end_dt)
            dfm5 = fetch_ohlcv(SYMBOL, "M5",  start_dt - _td(days=4),  end_dt)
            dfm30 = fetch_ohlcv(SYMBOL, "M30", start_dt - _td(days=45), end_dt)
        else:
            n_m1 = min(days * 1440 + 3000, 60000)
            df15 = fetch_latest(SYMBOL, "M15", count=days * 96 + 700)
            dfh1 = fetch_latest(SYMBOL, "H1", count=days * 24 + 400)
            dfh4 = fetch_latest(SYMBOL, "H4", count=days * 6 + 200)
            dfm1 = fetch_latest(SYMBOL, "M1", count=n_m1)
            dfm5 = fetch_latest(SYMBOL, "M5", count=days * 288 + 600)
            dfm30 = fetch_latest(SYMBOL, "M30", count=days * 48 + 300)
        for name, d in [("M15", df15), ("H1", dfh1), ("H4", dfh4), ("M1", dfm1), ("M5", dfm5)]:
            if d.empty:
                raise RuntimeError(f"No {name} data from MT5.")

        # ── Precompute features + GBM probs (batch, training convention) ─────
        aligned = align_to_primary({"M15": df15, "H1": dfh1, "H4": dfh4}, "M15")
        feats = build_live_features(aligned, feature_cols, "M15")
        if feats is None or feats.empty:
            raise RuntimeError("Feature precompute failed.")
        probs = pd.DataFrame({
            "buy":  stack.buy_specialist.predict_proba(feats),
            "sell": stack.sell_specialist.predict_proba(feats),
        }, index=feats.index)
        atr15 = ewm_atr(df15)
        atr_m30 = ewm_atr(dfm30) if not dfm30.empty else pd.Series(dtype=float)

        # ── Pick replay days: last N M1 dates with a real session ────────────
        m1_dates = dfm1.groupby(dfm1.index.date).size()
        valid_dates = [d for d, n in m1_dates.items() if n >= 300]
        feat_start_date = feats.index[50].date()
        valid_dates = [d for d in valid_dates if d > feat_start_date]
        if range_mode:
            lo = start_dt.date()
            hi = (end_dt - _td(days=1)).date()
            replay_dates = [d for d in valid_dates if lo <= d <= hi]
        else:
            replay_dates = valid_dates[-days:]
        if not replay_dates:
            raise RuntimeError("Not enough M1 history for the requested range.")
        replay_start = pd.Timestamp(replay_dates[0], tz="UTC")
        m1_replay = dfm1[dfm1.index >= replay_start]

        log("=" * 78)
        log(f"SYNQORA DELTA GOLD FABLE — RANGE REPLAY "
            f"({'guards ENFORCED' if enforce_guards else 'guards ADVISORY'})")
        log(f"Days: {len(replay_dates)}  ({replay_dates[0]} -> {replay_dates[-1]})  "
            f"| M1 bars: {len(m1_replay)}")
        log("=" * 78)

        # ── State ─────────────────────────────────────────────────────────────
        queue = SignalQueue()
        meta = MetaAgent()
        bar_no = 0
        open_trades, done_trades = [], []
        day_stats = {d: {"signals": 0, "releases": 0, "blocks": 0} for d in replay_dates}
        last_enqueue = {"BUY": None, "SELL": None}
        cur_day, daily_realized, day_halt = None, 0.0, False
        cur_regime = None   # latest M15 regime, passed to release guards
        m15_closes = set(df15.index + pd.Timedelta(minutes=15))

        for t_open in m1_replay.index:
            t = t_open + pd.Timedelta(minutes=1)
            day = t_open.date()
            if day not in day_stats:
                continue
            if day != cur_day:
                cur_day, daily_realized, day_halt = day, 0.0, False

            m1_closed_upto = dfm1[dfm1.index <= t_open]
            last_m1 = m1_closed_upto.iloc[-1]
            price = float(last_m1["close"])
            spread = float(last_m1.get("spread", 30.0))

            # ── Resolve open trades ────────────────────────────────────────
            still_open = []
            for tr in open_trades:
                hi, lo = float(last_m1["high"]), float(last_m1["low"])
                d_ = 1 if tr["side"] == "BUY" else -1
                fav = (hi - tr["entry"]) if d_ == 1 else (tr["entry"] - lo)
                adv = (tr["entry"] - lo) if d_ == 1 else (hi - tr["entry"])
                if fav > tr["mfe"]:
                    tr["mfe"], tr["mfe_time"] = fav, t
                tr["mae"] = max(tr["mae"], adv)

                held_m15 = (t - tr["entry_time"]).total_seconds() / 60.0 / 15.0
                exit_price, outcome = None, None
                if d_ == 1:
                    if lo <= tr["sl"]: exit_price, outcome = tr["sl"], "SL"
                    elif hi >= tr["tp"]: exit_price, outcome = tr["tp"], "TP"
                else:
                    if hi >= tr["sl"]: exit_price, outcome = tr["sl"], "SL"
                    elif lo <= tr["tp"]: exit_price, outcome = tr["tp"], "TP"
                if outcome is None and held_m15 >= NEGATIVE_TIME_STOP_BARS \
                        and (price - tr["entry"]) * d_ < 0:
                    exit_price, outcome = price, "NEG_TIME_STOP"
                if outcome is None and held_m15 >= MAX_HOLD_BARS:
                    exit_price, outcome = price, "MAX_HOLD"

                # ── P17: Two-stage trailing ratchet + virtual lever ────────
                if outcome is None and RATCHET_ENABLED:
                    r_price = tr.get("r_price", TRIPLE_BARRIER_SL_ATR * tr.get("atr", 0.0))
                    if r_price > 0:
                        peak_r = tr["mfe"] / r_price
                        if peak_r >= RATCHET_ARM_AT_R and not tr.get("r_locked", False):
                            lock_sl = tr["entry"] + d_ * RATCHET_LOCK_AT_R * r_price
                            if (d_ == 1 and lock_sl > tr["sl"]) or (d_ == -1 and lock_sl < tr["sl"]):
                                tr["sl"] = lock_sl
                                tr["r_locked"] = True
                            if RATCHET_VIRTUAL_LEVER:
                                if (d_ == 1 and lo <= lock_sl) or (d_ == -1 and hi >= lock_sl):
                                    exit_price, outcome = lock_sl, "VLEV"

                if outcome:
                    move = (exit_price - tr["entry"]) * d_ - tr["spread_cost"]
                    lot_mult = float(tr.get("m30_lot_mult", 1.0))
                    pnl = move * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT * lot_mult
                    daily_realized += pnl
                    tr.update(exit_time=t, exit_price=exit_price, outcome=outcome, pnl=pnl)
                    done_trades.append(tr)
                    # P1: Record trade outcome for streak tracking
                    meta.record_trade_outcome(tr["side"], pnl)
                else:
                    still_open.append(tr)
            open_trades = still_open

            # ── Daily halt ──────────────────────────────────────────────────
            if not day_halt and (daily_realized <= DAILY_LOSS_LIMIT
                                 or daily_realized >= DAILY_PROFIT_TARGET):
                day_halt = True
                queue = SignalQueue()
            if day_halt:
                continue

            # ── M15 close -> signal generation ──────────────────────────────
            if t in m15_closes:
                t_bar = t - pd.Timedelta(minutes=15)
                if t_bar in probs.index:
                    bar_no += 1
                    d15_closed = df15[df15.index <= t_bar]
                    regime = router.get_regime(d15_closed.tail(300))
                    cur_regime = regime
                    p = probs.loc[t_bar]
                    gate_ok = regime["trade_ok"] and regime["confidence"] >= REGIME_MIN_CONFIDENCE
                    # Option 2: Market-structure breakout override when gate is closed
                    if not gate_ok and BREAKOUT_ENABLED:
                        try:
                            bo = detect_market_breakout(
                                lookback_bars=BREAKOUT_LOOKBACK_BARS,
                                vol_mult=BREAKOUT_VOLUME_MULT,
                                min_strength=BREAKOUT_MIN_STRENGTH,
                            )
                        except Exception as e:
                            bo = None
                            log(f"[BREAKOUT] detection failed: {e}")
                        if bo is not None:
                            gate_ok = True
                            log(f"[BREAKOUT] {bo['direction']} breakout "
                                f"str={bo['strength']:.2f} — overriding regime gate")
                    if gate_ok:
                        open_pos_df = pd.DataFrame([{"type": x["side"]} for x in open_trades]) \
                                      if open_trades else pd.DataFrame()
                        decision = meta.decide(
                            {"buy_prob": float(p["buy"]), "sell_prob": float(p["sell"])},
                            regime, open_pos_df, daily_realized,
                            current_bar=bar_no, cooldown_bars=COOLDOWN_BARS)
                        signal_action = decision.action if decision.action in ("BUY", "SELL") else None
                        # Option 3: Dual-path ADX+EMA signal when GBM fails
                        if signal_action is None and DUAL_PATH_ENABLED:
                            try:
                                signal_action = check_trend_structure(
                                    d15_closed.tail(200), adx_min=DUAL_PATH_ADX_MIN,
                                    max_atr_extension=DUAL_PATH_MAX_ATR_EXTENSION)
                            except Exception as e:
                                log(f"[DUAL_PATH] check failed: {e}")
                            if signal_action is not None:
                                log(f"[DUAL_PATH] {signal_action} ADX+EMA signal "
                                    f"(prob={DUAL_PATH_PROB_FALLBACK:.2f})")
                        # ── Option 4: M30 Trend-Context Modulator ────────────
                        m30_prob_factor = 1.0
                        m30_lot_mult = 1.0
                        if signal_action is not None and DIRECTION_FILTER_ENABLED:
                            try:
                                m30_dir = m30_direction_filter(
                                    dfm30.tail(300) if not dfm30.empty else pd.DataFrame(),
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
                            is_dual = decision.action not in ("BUY", "SELL")
                            family = "TREND_M15" if is_dual else "GBM_M15"
                            same_p = DUAL_PATH_PROB_FALLBACK if is_dual else \
                                     float(p["buy"] if signal_action == "BUY" else p["sell"])
                            opp_p  = 0.0 if is_dual else \
                                     float(p["sell"] if signal_action == "BUY" else p["buy"])
                            # Apply M30 probability modulation (affects A+ dual-entry)
                            if m30_prob_factor != 1.0:
                                same_p = min(1.0, same_p * m30_prob_factor)
                            qprice = price
                            le = last_enqueue[signal_action]
                            if le is None or (t - le) >= pd.Timedelta(minutes=15):
                                queue.enqueue(QueuedSignal(
                                    side=signal_action, family=family,
                                    source_cid=f"{family}-{signal_action}-{uuid.uuid4().hex[:6]}",
                                    queue_price=qprice, queue_time=t.to_pydatetime(),
                                    m15_atr=float(atr15.loc[t_bar]), queue_spread=spread,
                                    meta={"prob": same_p, "edge": same_p - opp_p,
                                          "buy_prob": same_p if signal_action == "BUY" else 0.0,
                                          "sell_prob": same_p if signal_action == "SELL" else 0.0,
                                          "m30_lot_mult": m30_lot_mult,
                                          "regime": regime["regime"],
                                          "regime_conf": regime["confidence"]}))
                                last_enqueue[signal_action] = t
                                day_stats[day]["signals"] += 1

            # ── M1 release cycle ─────────────────────────────────────────────
            if len(queue) > 0:
                d15_g = df15[df15.index <= t - pd.Timedelta(minutes=15)].tail(200)
                dh4_g = dfh4[dfh4.index <= t - pd.Timedelta(minutes=240)].tail(40)
                dm5_g = dfm5[dfm5.index <= t - pd.Timedelta(minutes=5)].tail(20)

                # P2: Regime string for queue release threshold
                regime_name = str(cur_regime.get("regime", "UNKNOWN")) if cur_regime else "UNKNOWN"

                def hard_block(sig, _t=t, _d15=d15_g, _dh4=dh4_g, _p=price,
                               _day=day, _rg=cur_regime):
                    r = hard_block_reason(sig.side, _d15, _dh4, _p, sig.m15_atr,
                                          now=_t.to_pydatetime(), regime=_rg)
                    if r:
                        day_stats[_day]["blocks"] += 1
                    return r if enforce_guards else None

                released = queue.release_cycle(
                    df_m1=m1_closed_upto.tail(80), df_m5=dm5_g,
                    current_price=price, current_spread=spread,
                    hard_block_fn=hard_block, now=t.to_pydatetime(),
                    regime=regime_name)

                # P3: Fresh ATR at fill time if enabled
                if USE_FRESH_ATR_GEOMETRY and len(released) > 0:
                    d15_up_to = df15[df15.index <= t - pd.Timedelta(minutes=15)]
                    if len(d15_up_to) >= 15:
                        hl_f = d15_up_to["high"] - d15_up_to["low"]
                        hc_f = (d15_up_to["high"] - d15_up_to["close"].shift(1)).abs()
                        lc_f = (d15_up_to["low"] - d15_up_to["close"].shift(1)).abs()
                        tr_f = pd.concat([hl_f, hc_f, lc_f], axis=1).max(axis=1)
                        current_atr = float(tr_f.ewm(span=14, adjust=False).mean().iloc[-1])
                    else:
                        current_atr = released[0]["signal"].m15_atr if released else 0.0
                else:
                    current_atr = None

                for item in released:
                    sig = item["signal"]
                    dirn = 1 if sig.side == "BUY" else -1
                    entry = price
                    fill_atr = current_atr if USE_FRESH_ATR_GEOMETRY and current_atr else sig.m15_atr

                    # P11 auto: compute volatile flag at fill time
                    volatile_flag = False
                    if MFE_TRAIL_AUTO_MODE and fill_atr > 0:
                        _d15_recent = df15[df15.index <= t - pd.Timedelta(minutes=15)]
                        if len(_d15_recent) >= MFE_TRAIL_ATR_LOOKBACK + 20:
                            _atr_series = ewm_atr(_d15_recent.tail(MFE_TRAIL_ATR_LOOKBACK + 30))
                            _median_atr = float(_atr_series.tail(MFE_TRAIL_ATR_LOOKBACK).median())
                            if _median_atr > 0 and fill_atr / _median_atr >= MFE_TRAIL_ATR_RATIO:
                                volatile_flag = True
                        if not volatile_flag and cur_regime:
                            _reg = str(cur_regime.get("regime", ""))
                            if _reg in MFE_TRAIL_REGIME_TRIGGERS:
                                volatile_flag = True

                    open_trades.append({
                        "day": day, "side": sig.side, "cid": sig.source_cid,
                        "entry": entry,
                        "sl": entry - dirn * TRIPLE_BARRIER_SL_ATR * fill_atr,
                        "tp": entry + dirn * TRIPLE_BARRIER_TP_ATR * fill_atr,
                        "entry_time": t, "score": item["score"],
                        "spread_cost": spread * POINT,
                        "queue_price": sig.queue_price,
                        "queue_time": pd.Timestamp(sig.queue_time),
                        "r_price": TRIPLE_BARRIER_SL_ATR * fill_atr,
                        "atr": fill_atr,
                        "prob": float(sig.meta.get("prob", 0.0)),
                        "prob_edge": float(sig.meta.get("edge", 0.0)),
                        "m30_lot_mult": float(sig.meta.get("m30_lot_mult", 1.0)),
                        "mfe": 0.0, "mae": 0.0, "mfe_time": None,
                    })
                    meta.record_executed_signal(sig.side, bar_no)
                    day_stats[day]["releases"] += 1

        # Finalize trades still open at the very end.
        last_price = float(m1_replay["close"].iloc[-1])
        for tr in open_trades:
            d_ = 1 if tr["side"] == "BUY" else -1
            move = (last_price - tr["entry"]) * d_ - tr["spread_cost"]
            lot_mult = float(tr.get("m30_lot_mult", 1.0))
            tr.update(exit_time=m1_replay.index[-1], exit_price=last_price,
                      outcome="OPEN_END", pnl=move * LOT_SIZE * USD_PER_PRICE_UNIT_PER_LOT * lot_mult)
            done_trades.append(tr)

        # ══ REPORT ════════════════════════════════════════════════════════════
        variants = [("BE@0.3R", 0.3, "BE"), ("BE@0.5R", 0.5, "BE"), ("BE@0.7R", 0.7, "BE"), ("BE@1.0R", 1.0, "BE"),
                    ("LOCK60@1R", 1.0, "LOCK60"), ("TRAIL_11", 0.0, "TRAIL")]
        # P12: static target variants (P12 runs on regime-dependent targets 3/5/12,
        # so we test a range to bracket the effect)
        p12_variants = [(f"P12${t}", float(t), "P12") for t in [3, 5, 8, 12]]
        all_variants = variants + p12_variants
        for tr in done_trades:
            for name, arm, mode in all_variants:
                tr[name] = simulate_protection(tr, dfm1, arm, mode)["pnl"]
            d_ = 1 if tr["side"] == "BUY" else -1
            tr["chase_atr"] = (tr["entry"] - tr["queue_price"]) * d_ / tr["atr"]

        header_cols = (f"{'date':>10} {'sig':>4} {'rel':>4} {'blk':>5} {'trades':>6} {'wins':>5} "
                       f"{'RAW$':>8} {'BE@0.3R':>8} {'BE@0.5R':>8} {'BE@0.7R':>8} {'BE@1R':>8} {'LOCK60':>8} {'TRAIL':>8}")
        for name, _, _ in p12_variants:
            header_cols += f" {name:>8}"
        log("\n--- PER-DAY SUMMARY " + "-" * 58)
        log(header_cols)
        for d in replay_dates:
            trs = [t_ for t_ in done_trades if t_["day"] == d]
            wins = sum(1 for t_ in trs if t_["pnl"] > 0)
            raw = sum(t_["pnl"] for t_ in trs)
            row = " ".join([
                f"{str(d):>10}", f"{day_stats[d]['signals']:>4}",
                f"{day_stats[d]['releases']:>4}", f"{day_stats[d]['blocks']:>5}",
                f"{len(trs):>6}", f"{wins:>5}", f"{raw:>+8.2f}",
                f"{sum(t_['BE@0.3R'] for t_ in trs):>+8.2f}",
                f"{sum(t_['BE@0.5R'] for t_ in trs):>+8.2f}",
                f"{sum(t_['BE@0.7R'] for t_ in trs):>+8.2f}",
                f"{sum(t_['BE@1.0R'] for t_ in trs):>+8.2f}",
                f"{sum(t_['LOCK60@1R'] for t_ in trs):>+8.2f}",
                f"{sum(t_['TRAIL_11'] for t_ in trs):>+8.2f}",
            ])
            for name, _, _ in p12_variants:
                row += f" {sum(t_[name] for t_ in trs):>+8.2f}"
            log(row)

        n = len(done_trades)
        raw_total = sum(t_["pnl"] for t_ in done_trades)
        log(f"  (trend-regime exemptions suppressed "
            f"{entry_guards.trend_exemption_count} would-be blocks)")
        sep = "-" * (105 + len(p12_variants) * 9)
        log(sep)
        total_row = (f"{'TOTAL':>10} {sum(s['signals'] for s in day_stats.values()):>4} "
            f"{sum(s['releases'] for s in day_stats.values()):>4} "
            f"{sum(s['blocks'] for s in day_stats.values()):>5} {n:>6} "
            f"{sum(1 for t_ in done_trades if t_['pnl'] > 0):>5} {raw_total:>+8.2f} "
            f"{sum(t_['BE@0.3R'] for t_ in done_trades):>+8.2f} "
            f"{sum(t_['BE@0.5R'] for t_ in done_trades):>+8.2f} "
            f"{sum(t_['BE@0.7R'] for t_ in done_trades):>+8.2f} "
            f"{sum(t_['BE@1.0R'] for t_ in done_trades):>+8.2f} "
            f"{sum(t_['LOCK60@1R'] for t_ in done_trades):>+8.2f} "
            f"{sum(t_['TRAIL_11'] for t_ in done_trades):>+8.2f}")
        for name, _, _ in p12_variants:
            total_row += f" {sum(t_[name] for t_ in done_trades):>+8.2f}"
        log(total_row)

        if done_trades:
            log("\n--- OUTCOME BREAKDOWN " + "-" * 56)
            for oc in ("TP", "SL", "NEG_TIME_STOP", "MAX_HOLD", "OPEN_END"):
                sub = [t_ for t_ in done_trades if t_["outcome"] == oc]
                if sub:
                    log(f"  {oc:>14}: {len(sub):>3} trades | pnl {sum(t_['pnl'] for t_ in sub):>+9.2f} "
                        f"| avg MFE {np.mean([t_['mfe'] / t_['r_price'] for t_ in sub]):>5.2f}R")

            losers = [t_ for t_ in done_trades if t_["outcome"] in ("SL", "NEG_TIME_STOP")]
            if losers:
                log("\n--- LOSERS: HOW GREEN BEFORE DYING? " + "-" * 42)
                mfes = [t_["mfe"] / t_["r_price"] for t_ in losers]
                log(f"  losers: {len(losers)} | green net of spread: "
                    f"{sum(1 for t_ in losers if t_['mfe'] > t_['spread_cost'])} "
                    f"| >=+0.3R: {sum(1 for m in mfes if m >= 0.3)} "
                    f"| >=+0.5R: {sum(1 for m in mfes if m >= 0.5)} "
                    f"| >=+0.7R: {sum(1 for m in mfes if m >= 0.7)} "
                    f"| >=+1.0R: {sum(1 for m in mfes if m >= 1.0)}")
                log(f"  loser MFE: mean {np.mean(mfes):.2f}R | median {np.median(mfes):.2f}R "
                    f"| max {np.max(mfes):.2f}R")

            winners = [t_ for t_ in done_trades if t_["outcome"] == "TP"]
            if winners:
                scr05 = sum(1 for t_ in winners if t_["BE@0.5R"] < 1.0)
                scr10 = sum(1 for t_ in winners if t_["BE@1.0R"] < 1.0)
                log(f"  TP winners that BE@0.5R would have scratched: {scr05}/{len(winners)} "
                    f"| BE@1.0R: {scr10}/{len(winners)}")

            log("\n--- CHASE ANALYSIS (entry vs queue price, ATR units) " + "-" * 24)
            chases = [t_["chase_atr"] for t_ in done_trades]
            log(f"  mean {np.mean(chases):+.2f} ATR | median {np.median(chases):+.2f} "
                f"| improved entries: {sum(1 for c in chases if c < 0)}/{n}")
            for cap in (0.5, 1.0):
                kept = [t_ for t_ in done_trades if t_["chase_atr"] <= cap]
                excl = n - len(kept)
                log(f"  max-chase {cap:.1f} ATR: excludes {excl:>2} trades | RAW total "
                    f"{sum(t_['pnl'] for t_ in kept):>+9.2f} "
                    f"(vs {raw_total:+.2f}) | BE@0.5R total "
                    f"{sum(t_['BE@0.5R'] for t_ in kept):>+9.2f}")

            # P11: how many losers would TRAIL have saved?
            trail_saved = sum(1 for t_ in losers if t_["TRAIL_11"] > 0 and t_["pnl"] < 0)
            if losers:
                log(f"\n--- P11 TRAIL ANALYSIS " + "-" * 56)
                log(f"  TRAIL would convert {trail_saved}/{len(losers)} losers to small wins or scratch")
                trail_mfes = [t_["mfe"] / t_["r_price"] for t_ in losers if t_["TRAIL_11"] > t_["pnl"]]
                if trail_mfes:
                    log(f"  Saved loser MFE: mean {np.mean(trail_mfes):.2f}R")

            ledger = pd.DataFrame([{
                "day": t_["day"], "queue_time": t_["queue_time"],
                "entry_time": t_["entry_time"], "side": t_["side"],
                "score": t_["score"], "queue_price": t_["queue_price"],
                "entry": t_["entry"], "sl": t_["sl"], "tp": t_["tp"],
                "outcome": t_["outcome"], "exit_price": t_.get("exit_price"),
                "exit_time": t_.get("exit_time"),
                "r_price": t_["r_price"], "atr": t_["atr"],
                "prob": t_["prob"], "prob_edge": t_["prob_edge"],
                "mfe_r": t_["mfe"] / t_["r_price"], "mae_r": t_["mae"] / t_["r_price"],
                "chase_atr": t_["chase_atr"], "pnl_usd": t_["pnl"],
                "be05_usd": t_["BE@0.5R"], "be10_usd": t_["BE@1.0R"],
                "lock60_usd": t_["LOCK60@1R"], "trail11_usd": t_["TRAIL_11"],
            } for t_ in done_trades])
            tag = "guards" if enforce_guards else "noguards"
            path = out_path or os.path.join(
                "logs", f"replay_range_{replay_dates[0]}_{replay_dates[-1]}_{tag}.csv")
            ledger.to_csv(path, index=False)
            log(f"\n  Ledger saved: {path}")
        else:
            log("\n  No trades in the whole range.")
        log("=" * 78)

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
