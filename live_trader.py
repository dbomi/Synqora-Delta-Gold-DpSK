"""
=============================================================================
LIVE TRADER — SYNQORA DELTA GOLD FABLE
Clean orchestrator wiring the three validated pillars together:

  M15 bar close ──► features ──► GBM BUY/SELL probs ──► HMM regime gate
        ──► meta gate (prob ≥ 0.60, edge ≥ 0.15) ──► SIGNAL QUEUE (20, FIFO)

  M1 bar close ───► expire stale ──► leading-indicator scoring (per signal)
        ──► hard blocks (news / intraday extreme / H4 topzone)
        ──► release up to 3 per cycle (direction-balanced) ──► market order
            with SL=1.0×ATR(M15), TP=2.0×ATR(M15) from the signal's m15_atr

Signals are NEVER executed at generation time — the queue gate is the only
path to execution.
=============================================================================
"""

import logging
import os
import time
import uuid
from datetime import datetime, timezone

from config import (
    SETUP_TAG, SETUP_VERSION, SYMBOL, PRIMARY_TF, CONTEXT_TFS,
    FAST_TF, QUEUE_ALIGN_TF, LOT_SIZE,
    MODELS_DIR, LOGS_DIR, COOLDOWN_BARS, MAX_POSITIONS,
    REGIME_MIN_CONFIDENCE,
    QUEUE_DEDUP_SAME_SIDE_BARS,
    DAILY_PROFIT_TARGET, DAILY_LOSS_LIMIT,
    REQUIRE_DEMO_ACCOUNT, ALLOW_REAL_ACCOUNT, EXECUTE_TRADES,
    HEARTBEAT_MINUTES,
    META_THRESHOLDS, META_MIN_PROB_EDGE,
    QUEUE_CAPACITY, QUEUE_RELEASE_SCORE, QUEUE_MAX_PENDING_MINUTES,
    LOT_SIZING_MODE, RISK_PCT_PER_TRADE, CAMPAIGN_MAX_LOT,
    A_PLUS_PROB_THRESHOLD, A_PLUS_MIN_EQUITY,
    GUARD_TREND_REGIME_EXEMPTION, GUARD_EXEMPT_MIN_REGIME_CONF,
    USE_EQUITY_TIERED_BREAKEVEN, BREAKEVEN_EQUITY_CUTOFF,
)
from data_engine import (
    initialize_mt5, shutdown_mt5, fetch_multi_tf_latest, align_to_primary,
    fetch_latest, is_new_bar, get_daily_pnl,
)
from feature_engine import build_live_features
from model_stack import ModelStack
from regime_detector import RegimeRouter
from meta_agent import MetaAgent
from signal_queue import SignalQueue, QueuedSignal
from entry_guards import hard_block_reason
from execution_engine import place_market_order, get_current_spread_points
from position_manager import manage_open_positions, open_position_counts, open_positions_frame

import MetaTrader5 as mt5
import pandas as pd
import pickle

# ── Logging setup: per-session file under logs/sessions/ ───────────────────
SESSIONS_DIR = os.path.join(LOGS_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)
SESSION_LOG = os.path.join(SESSIONS_DIR, f"fable_live_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(name)-16s | %(levelname)-7s | %(message)s",
    handlers= [
        logging.FileHandler(SESSION_LOG, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveTrader")

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}


def _drop_forming(df, tf_minutes: int):
    """
    Keep only CLOSED bars — the research replays always evaluated on closed
    bars, so live must too. A bar is closed when open_time + duration <= now.
    """
    if df is None or df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=tf_minutes) <= now]


class FableLiveTrader:

    def __init__(self):
        self.stack        = ModelStack()
        self.regime       = RegimeRouter()
        self.meta         = MetaAgent()
        self.queue        = SignalQueue()
        self.feature_cols = []

        self._last_primary_bar = None
        self._last_m1_bar      = None
        self._bar_counter      = 0
        self._session_date     = None
        self._daily_halt       = False
        self._last_regime      = None   # latest M15 regime, reused by release guards

    # ── Startup ────────────────────────────────────────────────────────────
    def load_models(self):
        self.stack.load(MODELS_DIR)
        self.regime.load(MODELS_DIR)
        with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
            self.feature_cols = pickle.load(f)
        logger.info(f"Models loaded: GBM buy/sell + HMM regime router | "
                    f"{len(self.feature_cols)} features")

    # ── Daily session handling ────────────────────────────────────────────
    def _roll_session_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if self._session_date != today:
            self._session_date = today
            self._daily_halt   = False
            self.meta.reset_session_counters()
            logger.info(f"New trading day {today}. Session counters reset.")

    def _daily_limits_hit(self) -> bool:
        if self._daily_halt:
            return True
        pnl = get_daily_pnl()
        if pnl >= DAILY_PROFIT_TARGET or pnl <= DAILY_LOSS_LIMIT:
            logger.warning(f"Daily P&L limit hit ({pnl:.2f}). Halting for the day; "
                           f"clearing {len(self.queue)} queued signals.")
            self._daily_halt = True
            self.queue = SignalQueue()   # drop all pending signals
            return True
        return False

    # ── SIGNAL GENERATION (per M15 close) ─────────────────────────────────
    def _generate_signal(self):
        data = fetch_multi_tf_latest(SYMBOL, PRIMARY_TF, CONTEXT_TFS, count=500)
        if PRIMARY_TF not in data or data[PRIMARY_TF].empty:
            logger.warning("No primary TF data. Skipping signal cycle.")
            return
        # Closed bars only, on every timeframe — matches the validated replays
        # (predicting on the forming bar would feed half-empty candles to the GBM).
        data = {tf: _drop_forming(df, TF_MINUTES.get(tf, 15)) for tf, df in data.items()}
        if data[PRIMARY_TF].empty:
            return
        data_aligned = align_to_primary(data, PRIMARY_TF)
        df_primary   = data_aligned[PRIMARY_TF]

        features = build_live_features(data_aligned, self.feature_cols, PRIMARY_TF)
        if features is None or features.empty:
            logger.warning("Feature build failed. Skipping signal cycle.")
            return

        # HMM regime intelligence gate
        regime_result = self.regime.get_regime(df_primary)
        self._last_regime = regime_result
        logger.info(f"[REGIME] {regime_result['regime']} "
                    f"conf={regime_result['confidence']:.2f} "
                    f"cusum_warning={regime_result['cusum_warning']} "
                    f"trade_ok={regime_result['trade_ok']}")
        if not regime_result["trade_ok"] or regime_result["confidence"] < REGIME_MIN_CONFIDENCE:
            logger.info("[REGIME] Gate closed — no signal generation this bar.")
            return

        # GBM model probabilities
        model_output = self.stack.predict(features)
        logger.info(f"[MODEL] buy_prob={model_output['buy_prob']:.3f} "
                    f"sell_prob={model_output['sell_prob']:.3f}")

        # Meta gate (validated thresholds + cooldown + position limits)
        decision = self.meta.decide(
            model_output   = model_output,
            regime_result  = regime_result,
            open_positions = open_positions_frame(),
            daily_pnl      = get_daily_pnl(),
            current_bar    = self._bar_counter,
            cooldown_bars  = COOLDOWN_BARS,
        )
        logger.info(f"[META] {decision.reason}")
        if decision.action not in ("BUY", "SELL"):
            return

        # Dedup: skip if same side+family already queued very recently
        dedup_minutes = QUEUE_DEDUP_SAME_SIDE_BARS * TF_MINUTES.get(PRIMARY_TF, 15)
        if self.queue.has_recent_same_signal(decision.action, "GBM_M15", dedup_minutes):
            logger.info(f"[QUEUE] Duplicate {decision.action} within "
                        f"{dedup_minutes}min already queued. Skipping enqueue.")
            return

        # Enqueue — never execute directly
        hl = df_primary["high"] - df_primary["low"]
        hc = (df_primary["high"] - df_primary["close"].shift(1)).abs()
        lc = (df_primary["low"]  - df_primary["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        m15_atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

        tick = mt5.symbol_info_tick(SYMBOL)
        queue_price = float(tick.ask if decision.action == "BUY" else tick.bid) \
                      if tick else float(df_primary["close"].iloc[-1])

        sig = QueuedSignal(
            side         = decision.action,
            family       = "GBM_M15",
            source_cid   = f"GBM-{decision.action}-{uuid.uuid4().hex[:8]}",
            queue_price  = queue_price,
            queue_time   = datetime.now(timezone.utc),
            m15_atr      = m15_atr,
            queue_spread = get_current_spread_points(SYMBOL),
            meta = {
                "buy_prob":  decision.buy_prob,
                "sell_prob": decision.sell_prob,
                "regime":     decision.regime,
                "regime_conf": regime_result["confidence"],
            },
        )
        self.queue.enqueue(sig)

    # ── QUEUE RELEASE (per M1 close) ───────────────────────────────────────
    def _release_cycle(self):
        if len(self.queue) == 0:
            return

        df_m1 = fetch_latest(SYMBOL, FAST_TF, count=80)
        df_m5 = fetch_latest(SYMBOL, QUEUE_ALIGN_TF, count=20)
        if df_m1.empty:
            return
        # Drop the forming bar — scoring uses CLOSED bars only.
        df_m1 = df_m1.iloc[:-1]
        df_m5 = df_m5.iloc[:-1] if not df_m5.empty else df_m5

        df_primary = fetch_latest(SYMBOL, PRIMARY_TF, count=120)
        df_h4      = fetch_latest(SYMBOL, "H4", count=40)
        if not df_h4.empty:
            df_h4 = df_h4.iloc[:-1]

        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return
        current_spread = get_current_spread_points(SYMBOL)

        def hard_block(sig: QueuedSignal):
            price = float(tick.ask if sig.side == "BUY" else tick.bid)
            return hard_block_reason(
                side          = sig.side,
                df_primary    = df_primary,
                df_h4         = df_h4,
                current_price = price,
                m15_atr       = sig.m15_atr,
                regime        = self._last_regime,
            )

        mid_price = (float(tick.ask) + float(tick.bid)) / 2.0
        released = self.queue.release_cycle(
            df_m1          = df_m1,
            df_m5          = df_m5,
            current_price  = mid_price,
            current_spread = current_spread,
            hard_block_fn  = hard_block,
        )

        for item in released:
            sig = item["signal"]
            counts = open_position_counts()
            if counts[sig.side] >= MAX_POSITIONS:
                logger.info(f"[EXEC] {sig.side} released but max positions "
                            f"({counts[sig.side]}) reached. Dropping {sig.source_cid}.")
                continue

            # Scale campaign: risk-percent sizing + A+ dual entry.
            prob_key = "buy_prob" if sig.side == "BUY" else "sell_prob"
            prob = float(sig.meta.get(prob_key, 0.0))
            try:
                from lot_campaign import get_signal_lots
                lot_each, n_pos = get_signal_lots(sig.side, prob, sig.m15_atr)
            except Exception as e:
                logger.warning(f"Campaign sizing failed ({e}); fallback 1x {LOT_SIZE}.")
                lot_each, n_pos = LOT_SIZE, 1
            if n_pos <= 0:
                logger.warning(f"[EXEC] {sig.source_cid} sized to zero "
                               f"(no equity/margin room). Dropping.")
                continue
            # Don't blow through the per-direction position cap with dual entry.
            n_pos = min(n_pos, MAX_POSITIONS - counts[sig.side])

            placed = 0
            for k in range(n_pos):
                suffix = f"|{k + 1}" if n_pos > 1 else ""
                result = place_market_order(
                    side    = sig.side,
                    volume  = lot_each,
                    atr     = sig.m15_atr,
                    comment = f"Fable|{sig.source_cid}{suffix}",
                )
                if result.success:
                    placed += 1
                    logger.info(f"[EXEC] {sig.side} {lot_each} lots "
                                f"({k + 1}/{n_pos}) cid={sig.source_cid} "
                                f"score={item['score']:.1f} prob={prob:.2f} "
                                f"ticket={result.ticket}")
                else:
                    logger.warning(f"[EXEC] Order {k + 1}/{n_pos} failed for "
                                   f"{sig.source_cid}: {result.error_msg}")
                    break
            if placed > 0:
                self.meta.record_executed_signal(sig.side, self._bar_counter)

    # ── Startup safety + preflight banner ──────────────────────────────────
    def _verify_demo_account(self):
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
        is_demo = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        logger.info(f"[MT5] {acc.name} @ {acc.server} | login={acc.login} | "
                    f"equity=${acc.equity:,.2f} {acc.currency} | "
                    f"trade_mode={'DEMO' if is_demo else 'REAL/CONTEST'}")
        if REQUIRE_DEMO_ACCOUNT and not is_demo and not ALLOW_REAL_ACCOUNT:
            raise RuntimeError(
                f"DEMO GUARD: account {acc.login} @ {acc.server} is not a demo "
                f"account. Refusing to trade (REQUIRE_DEMO_ACCOUNT=True).")
        term = mt5.terminal_info()
        if term is not None and not term.trade_allowed:
            logger.warning("[MT5] Algo Trading is OFF in the terminal — orders "
                           "will be rejected. Enable the 'Algo Trading' button.")
        return acc

    def _log_preflight(self):
        thr = META_THRESHOLDS["UNKNOWN"]
        logger.info("[PREFLIGHT] Configuration as validated in research:")
        logger.info(f"[PREFLIGHT]   symbol={SYMBOL} primary={PRIMARY_TF} "
                    f"execute_trades={EXECUTE_TRADES}")
        logger.info(f"[PREFLIGHT]   entry gate: prob>={thr['buy']} "
                    f"edge>={META_MIN_PROB_EDGE} regime_conf>={REGIME_MIN_CONFIDENCE}")
        logger.info(f"[PREFLIGHT]   queue: cap={QUEUE_CAPACITY} "
                    f"release_score>={QUEUE_RELEASE_SCORE} "
                    f"expiry={QUEUE_MAX_PENDING_MINUTES}min")
        logger.info(f"[PREFLIGHT]   guards: news+extreme+H4zone | trend exemption="
                    f"{GUARD_TREND_REGIME_EXEMPTION} (conf>={GUARD_EXEMPT_MIN_REGIME_CONF})")
        logger.info(f"[PREFLIGHT]   campaign: {LOT_SIZING_MODE} "
                    f"{RISK_PCT_PER_TRADE}%/trade cap={CAMPAIGN_MAX_LOT} | "
                    f"A+ x2 prob>={A_PLUS_PROB_THRESHOLD} equity>=${A_PLUS_MIN_EQUITY:.0f}")
        logger.info(f"[PREFLIGHT]   protection: BE@+1R below "
                    f"${BREAKEVEN_EQUITY_CUTOFF:.0f} ({USE_EQUITY_TIERED_BREAKEVEN}) | "
                    f"neg-time-stop 24 bars | max hold 48 | "
                    f"daily limits {DAILY_LOSS_LIMIT}/{DAILY_PROFIT_TARGET}")
        logger.info(f"[PREFLIGHT]   max positions/side={MAX_POSITIONS} | "
                    f"session log: {SESSION_LOG}")

    def _heartbeat(self):
        acc = mt5.account_info()
        if acc is None:
            logger.warning("[STATUS] no account info from MT5")
            return
        counts = open_position_counts()
        logger.info(f"[STATUS] equity=${acc.equity:,.2f} balance=${acc.balance:,.2f} "
                    f"| open BUY={counts['BUY']} SELL={counts['SELL']} "
                    f"| queue={len(self.queue)} | daily_pnl={get_daily_pnl():+,.2f} "
                    f"| halt={self._daily_halt}")

    # ── MAIN LOOP ──────────────────────────────────────────────────────────
    def run(self, poll_interval: float = 5.0):
        logger.info("="*60)
        logger.info(f"{SETUP_TAG} v{SETUP_VERSION} | DEMO-LIVE")
        logger.info("Pipeline: GBM(M15) -> HMM regime gate -> Signal Queue "
                    "-> M1 leading-indicator release -> risk-pct campaign")
        logger.info("="*60)

        if not initialize_mt5():
            raise RuntimeError("MT5 connection failed. Check terminal is open and logged in.")
        self._verify_demo_account()
        self.load_models()
        self._log_preflight()
        last_heartbeat = time.monotonic()

        try:
            while True:
                try:
                    self._roll_session_if_needed()

                    # Position management runs every cycle regardless of halts.
                    manage_open_positions()

                    if not self._daily_limits_hit():
                        # New M15 bar → signal generation
                        new_bar, bar_time = is_new_bar(SYMBOL, PRIMARY_TF, self._last_primary_bar)
                        if new_bar:
                            self._last_primary_bar = bar_time
                            self._bar_counter += 1
                            logger.info(f"── New {PRIMARY_TF} bar {bar_time} "
                                        f"(#{self._bar_counter}) | queue depth "
                                        f"{len(self.queue)} ──")
                            self._generate_signal()

                        # New M1 bar → queue release cycle
                        new_m1, m1_time = is_new_bar(SYMBOL, FAST_TF, self._last_m1_bar)
                        if new_m1:
                            self._last_m1_bar = m1_time
                            self.queue.expire_stale()
                            self._release_cycle()

                    if time.monotonic() - last_heartbeat >= HEARTBEAT_MINUTES * 60:
                        last_heartbeat = time.monotonic()
                        self._heartbeat()

                    time.sleep(poll_interval)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Main loop error: {e}", exc_info=True)
                    time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            shutdown_mt5()


if __name__ == "__main__":
    trader = FableLiveTrader()
    trader.run(poll_interval=5.0)
