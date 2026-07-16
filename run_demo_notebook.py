"""
Synqora Delta Gold Fable v1.1 — Single-Cell Notebook / VS Code Launcher
=========================================================================
Paste this entire file into ONE notebook cell and run it.

Requires:
  - MetaTrader 5 open, logged into your DEMO account
  - Algo Trading enabled in MT5 (toolbar button)
  - Python 3.11 with MetaTrader5, pandas, numpy, scikit-learn,
    lightgbm, xgboost, hmmlearn

The cell validates config + models + MT5 (with a hard demo-account guard),
prints the preflight summary, then launches live_trader.py as a subprocess
with filtered live log streaming. Trades ARE executed on the demo account.

Stop it with the notebook interrupt button (sends KeyboardInterrupt to the
runner, which shuts down MT5 cleanly).
"""

# ─── 0. Locate this script's directory ───────────────────────────────────────
from pathlib import Path
import os, sys, pickle, subprocess
from datetime import datetime

try:
    _nb = Path(globals().get("__vsc_ipynb_file__",
               globals().get("__file__", ".")))
    PROJECT_DIR = _nb.parent if _nb.is_file() else Path(os.getcwd())
except Exception:
    PROJECT_DIR = Path(os.getcwd())

print(f"PROJECT_DIR: {PROJECT_DIR}")
os.chdir(str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

# ─── 1. Config preflight — values must match the validated research ──────────
import config as cfg

assert cfg.REQUIRE_DEMO_ACCOUNT,                    "REQUIRE_DEMO_ACCOUNT must be True"
assert not cfg.ALLOW_REAL_ACCOUNT,                  "ALLOW_REAL_ACCOUNT must be False"
assert cfg.EXECUTE_TRADES,                          "EXECUTE_TRADES must be True (demo validation)"
assert cfg.SYMBOL,                                  "SYMBOL not set"
# Entry gate exactly as walk-forward validated
assert cfg.META_THRESHOLDS["UNKNOWN"]["buy"] == 0.60, "entry threshold drifted from 0.60"
assert cfg.META_MIN_PROB_EDGE == 0.15,              "prob edge drifted from 0.15"
# Label/execution geometry the GBMs were trained on
assert cfg.TRIPLE_BARRIER_TP_ATR == 2.0 and cfg.TRIPLE_BARRIER_SL_ATR == 1.0, \
    "SL/TP geometry drifted from the trained 2.0/1.0 — retrain required if intended"
# Queue gate as tested
assert cfg.QUEUE_CAPACITY == 20 and cfg.QUEUE_RELEASE_SCORE == 4.0, "queue params drifted"
assert cfg.QUEUE_MAX_RELEASES_PER_CYCLE == 3 and cfg.QUEUE_MAX_RELEASES_PER_SIDE == 2
# Scale campaign as selected in the semester test
assert cfg.LOT_SIZING_MODE == "RISK_PCT" and cfg.RISK_PCT_PER_TRADE == 6.0
assert cfg.CAMPAIGN_MAX_LOT == 5.0 and cfg.MAX_TOTAL_SIGNAL_LOT == 5.0
assert cfg.A_PLUS_DUAL_ENTRY and cfg.A_PLUS_PROB_THRESHOLD == 0.75
assert cfg.A_PLUS_MIN_EQUITY == 1000.0,             "A+ equity gate drifted from $1k"
# P5 always-on BE
assert cfg.BREAKEVEN_ALWAYS_ON, "P5 BE always-on required"
# P11 MFE trail
assert cfg.MFE_TRAIL_ENABLED, "P11 MFE trail required"
# P14 regime protection with virtual stop
assert cfg.REGIME_PROTECTION_ENABLED, "P14 regime protection required"
# P13 golden-hour campaign
assert cfg.GOLDEN_HOUR_CAMPAIGN_ENABLED
assert set(cfg.GOLDEN_HOURS_UTC) == {2, 5, 6, 16, 17, 18}, "golden hours drifted"
assert cfg.GOLDEN_HOUR_LOT_MULT == 2.5, "golden hour mult changed"
assert cfg.GOLDEN_HOUR_A_PLUS_THRESHOLD == 0.60
assert set(cfg.DEAD_HOURS_UTC) == {9, 11}, "dead hours changed"
assert cfg.DEAD_HOUR_LOT_MULT == 0.3, "dead hour mult changed"
# P12 must be off (proven destructive)
assert not cfg.POSITION_MONITOR_ENABLED, "P12 must be off"
# Guard softening + survival protections
assert cfg.GUARD_TREND_REGIME_EXEMPTION and cfg.GUARD_EXEMPT_MIN_REGIME_CONF == 0.70
assert cfg.USE_EQUITY_TIERED_BREAKEVEN or cfg.BREAKEVEN_ALWAYS_ON
assert cfg.MAX_POSITIONS == 6
# P2 regime-conditional score
assert cfg.QUEUE_RELEASE_SCORE_BY_REGIME["TREND_UP"] == 3.5
assert cfg.QUEUE_RELEASE_SCORE_BY_REGIME["VOLATILE"] == 5.5

print("[PREFLIGHT] Config OK")
print(f"  setup:            {cfg.SETUP_TAG} v{cfg.SETUP_VERSION}")
print(f"  symbol / magic:   {cfg.SYMBOL} / {cfg.MAGIC_NUMBER}")
print(f"  entry gate:       prob>={cfg.META_THRESHOLDS['UNKNOWN']['buy']} "
      f"edge>={cfg.META_MIN_PROB_EDGE} regime_conf>={cfg.REGIME_MIN_CONFIDENCE}")
print(f"  queue:            cap={cfg.QUEUE_CAPACITY} regime-score "
      f"3.5/4.5/5.0/5.5 anti-chase cluster-gate")
print(f"  campaign:         RISK {cfg.RISK_PCT_PER_TRADE}%/trade -> cap {cfg.CAMPAIGN_MAX_LOT} lots"
      f" | A+ x{cfg.A_PLUS_POSITION_COUNT} at prob>={cfg.A_PLUS_PROB_THRESHOLD}"
      f" once equity>=${cfg.A_PLUS_MIN_EQUITY:.0f}")
print(f"  P13 golden hours: {cfg.GOLDEN_HOURS_UTC} x{cfg.GOLDEN_HOUR_LOT_MULT} lot"
      f" + A+ threshold {cfg.GOLDEN_HOUR_A_PLUS_THRESHOLD}"
      f" | dead hours {cfg.DEAD_HOURS_UTC} x{cfg.DEAD_HOUR_LOT_MULT}")
print(f"  protection:       P5 BE@1.0R always-on | P11 MFE trail (auto) | "
      f"P14 regime ladder + virtual stop | "
      f"neg-time-stop {cfg.NEGATIVE_TIME_STOP_BARS} | max hold {cfg.MAX_HOLD_BARS}")
print(f"  daily limits:     {cfg.DAILY_LOSS_LIMIT:+.0f} / {cfg.DAILY_PROFIT_TARGET:+.0f} USD")
print(f"  P1 streak:        reduce after {cfg.CONSECUTIVE_LOSS_REDUCE_AFTER}L "
      f"x{cfg.CONSECUTIVE_LOSS_REDUCE_BY}, boost after {cfg.CONSECUTIVE_WIN_INCREASE_AFTER}W "
      f"x{cfg.CONSECUTIVE_WIN_INCREASE_BY}")

# ─── 2. Model preflight ──────────────────────────────────────────────────────
MODELS = Path(cfg.MODELS_DIR)
RUNNER = PROJECT_DIR / "live_trader.py"
for label, path in {
    "GBM_BUY":      MODELS / "gbm_buy.pkl",
    "GBM_SELL":     MODELS / "gbm_sell.pkl",
    "HMM_REGIME":   MODELS / "hmm_regime.pkl",
    "CUSUM":        MODELS / "cusum_state.pkl",
    "FEATURE_COLS": MODELS / "feature_cols.pkl",
    "RUNNER":       RUNNER,
}.items():
    assert path.exists(), f"MISSING: {label} -> {path}"
    print(f"  [OK] {label}: {path.name}")

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from model_stack import ModelStack
from regime_detector import RegimeRouter, TrendRegimeDetector

stack = ModelStack().load(str(MODELS))
with open(MODELS / "feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)
assert len(feature_cols) == 76, f"expected 76 features, got {len(feature_cols)}"
smoke = stack.predict(pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols))
assert 0.0 <= smoke["buy_prob"] <= 1.0 and 0.0 <= smoke["sell_prob"] <= 1.0
print(f"  [OK] GBM smoke predict: buy={smoke['buy_prob']:.3f} sell={smoke['sell_prob']:.3f}")

from config import REGIME_USE_TREND_DETECTOR
router_cls = TrendRegimeDetector if REGIME_USE_TREND_DETECTOR else RegimeRouter
router = router_cls().load(str(MODELS))
if hasattr(router, 'hmm'):
    assert set(router.hmm.state_map.values()) == \
        {"TREND_UP", "TREND_DOWN", "RANGING", "VOLATILE", "FLAT"}
    print(f"  [OK] HMM regime router (retrained): {router.hmm.state_map}")
else:
    reg = router.get_regime(pd.DataFrame({'high': [4100], 'low': [4090], 'close': [4095]}))
    assert reg['regime'] in ('TREND_UP', 'TREND_DOWN', 'RANGING', 'VOLATILE', 'FLAT', 'UNKNOWN')
    print(f"  [OK] Trend regime detector: get_regime works ({reg['regime']})")

from signal_queue import SignalQueue
from entry_guards import hard_block_reason
from lot_campaign import compute_signal_lots
from position_manager import start_regime_protection
lot, n = compute_signal_lots(equity=1500, m15_atr=5.5, prob=0.80)
assert n == 2, "A+ dual entry not active above $1k in sizing math"
print(f"  [OK] Queue gate, hard-block guards, campaign sizing "
      f"(A+ sample: {n} x {lot} lots @ $1.5k; golden hour x{cfg.GOLDEN_HOUR_LOT_MULT})")

# ─── 3. MT5 preflight — hard demo guard ──────────────────────────────────────
import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
acct = mt5.account_info()
assert acct is not None, f"account_info() failed: {mt5.last_error()}"
is_demo = acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
assert is_demo, (f"DEMO GUARD: {acct.login} @ {acct.server} is not a demo account "
                 f"(trade_mode={acct.trade_mode}). Aborting.")
term = mt5.terminal_info()
algo_on = bool(term and term.trade_allowed)
sym = mt5.symbol_info(cfg.SYMBOL)
assert sym is not None, f"Symbol {cfg.SYMBOL} not found at this broker"
if not sym.visible:
    mt5.symbol_select(cfg.SYMBOL, True)
print(f"\n[MT5] Connected — {acct.name} @ {acct.server}")
print(f"  account:      {acct.login}  (DEMO)")
print(f"  equity:       ${acct.equity:,.2f} {acct.currency}")
print(f"  leverage:     1:{acct.leverage}")
print(f"  algo trading: {'ON' if algo_on else '*** OFF — ENABLE IT OR ORDERS WILL FAIL ***'}")
print(f"  symbol:       {cfg.SYMBOL} spread={sym.spread}pts "
      f"vol {sym.volume_min}-{sym.volume_max} step {sym.volume_step}")
mt5.shutdown()

# ─── 4. Behaviour summary ────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("PIPELINE BEHAVIOUR (exactly as replay-validated Jan-Jun 2026)")
print("=" * 62)
print(f"""
 M15 close -> 76 features -> GBM buy/sell probs
    -> HMM regime gate (conf>=0.50, trade_ok)
    -> meta gate (prob>=0.60, edge>=0.15, cooldown, position caps)
    -> SIGNAL QUEUE (never executed directly)

 M1 close  -> expire stale (15min chased / 90min normal)
    -> P2 regime-conditional score threshold
       (trend=3.5, ranging=5.0, volatile=5.5, flat=4.5)
    -> anti-chase guard (entry at-or-better than queue price +0.15ATR)
    -> hard blocks: news / intraday extreme / H4 zone / session-open
       (extreme+zone skipped when HMM confidently agrees with side)
    -> cluster gate (<=2/same-side per 15min window)
    -> release best-first (<=3/cycle, <=2/side)
    -> P13 golden-hour campaign: {cfg.GOLDEN_HOURS_UTC} x{cfg.GOLDEN_HOUR_LOT_MULT} lot
       (A+ threshold {cfg.GOLDEN_HOUR_A_PLUS_THRESHOLD}); {cfg.DEAD_HOURS_UTC} x{cfg.DEAD_HOUR_LOT_MULT}
    -> P1 streak sizing: reduce after {cfg.CONSECUTIVE_LOSS_REDUCE_AFTER}L x{cfg.CONSECUTIVE_LOSS_REDUCE_BY},
       boost after {cfg.CONSECUTIVE_WIN_INCREASE_AFTER}W x{cfg.CONSECUTIVE_WIN_INCREASE_BY}
    -> risk-pct sizing ({cfg.RISK_PCT_PER_TRADE}%/trade)
    -> A+ dual entry at prob>={cfg.A_PLUS_PROB_THRESHOLD} once equity>=${cfg.A_PLUS_MIN_EQUITY:.0f}
    -> P3 fresh-ATR SL={cfg.TRIPLE_BARRIER_SL_ATR}xATR / TP={cfg.TRIPLE_BARRIER_TP_ATR}xATR

 P14   -> Regime-laddered profit lock (VOLATILE/TRENDING/RANGING)
         -> virtual stop enforces exit if broker SL modify fails
         -> locks profit at +0.15R / +0.5R / +1.0R steps per regime
         -> once green, NEVER red
 Always -> P5 BE@1.0R always-on (every trade)
         -> P11 MFE trail (auto in volatile/high-ATR)
         -> P7 directional loss limiter (50% of daily limit per side)
         -> P10 margin validation (min 10% free margin after trade)
         -> neg-time-stop {cfg.NEGATIVE_TIME_STOP_BARS} bars | max hold {cfg.MAX_HOLD_BARS}
         -> daily halt at {cfg.DAILY_LOSS_LIMIT:+.0f}/{cfg.DAILY_PROFIT_TARGET:+.0f}
         -> [STATUS] heartbeat every {cfg.HEARTBEAT_MINUTES} min

 Expect few trades per day; queue depth and block reasons appear in
 the log. Golden-hour sizing (x{cfg.GOLDEN_HOUR_LOT_MULT}) active in UTC hours
 {cfg.GOLDEN_HOURS_UTC}. First A+ dual entries appear above ${cfg.A_PLUS_MIN_EQUITY:.0f}.
 """)

# ─── 5. Launch runner with filtered log streaming ────────────────────────────
IMPORTANT_PATTERNS = (
    "Synqora", "DEMO-LIVE", "[PREFLIGHT]", "[MT5]", "[REGIME]", "[MODEL]",
    "[META]", "[QUEUE]", "[GUARD]", "[CAMPAIGN]", "[EXEC]", "[PM]",
    "[STATUS]", "[NEWS]", "New trading day", "Daily P&L limit",
    "ERROR", "Traceback", "Models loaded",
)
# High-noise lines hidden from the notebook (kept in the session log file):
NOISE_PATTERNS = ("NO_TRADE:",)

print("=" * 62)
print("Starting Fable demo-live runner...  (interrupt the cell to stop)")
print("=" * 62 + "\n")

nb_log = PROJECT_DIR / "logs" / "sessions" / \
    f"fable_notebook_{datetime.now():%Y%m%d_%H%M%S}.log"
nb_log.parent.mkdir(parents=True, exist_ok=True)
print(f"Notebook mirror log: {nb_log}")
print(f"(runner writes its own timestamped log to logs/sessions/ as well)\n")

env = dict(os.environ, PYTHONIOENCODING="utf-8")
proc = subprocess.Popen(
    [sys.executable, str(RUNNER)],
    cwd=str(PROJECT_DIR),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, encoding="utf-8", errors="replace", env=env)

assert proc.stdout is not None
try:
    with nb_log.open("a", encoding="utf-8") as lf:
        for line in proc.stdout:
            lf.write(line); lf.flush()
            if any(p in line for p in NOISE_PATTERNS):
                continue
            if any(p in line for p in IMPORTANT_PATTERNS):
                print(line, end="", flush=True)
except KeyboardInterrupt:
    print("\n[NOTEBOOK] Interrupt received — stopping runner...")
    proc.terminate()
ret = proc.wait()
print(f"\nFable runner exited with code: {ret}")
