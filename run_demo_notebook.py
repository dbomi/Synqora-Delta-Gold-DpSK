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
assert cfg.LOT_SIZING_MODE == "RISK_PCT" and cfg.RISK_PCT_PER_TRADE == 2.0
assert cfg.CAMPAIGN_MAX_LOT == 5.0 and cfg.MAX_TOTAL_SIGNAL_LOT == 5.0
assert cfg.A_PLUS_DUAL_ENTRY and cfg.A_PLUS_PROB_THRESHOLD == 0.70
assert cfg.A_PLUS_MIN_EQUITY == 1000.0,             "A+ equity gate drifted from $1k"
# Guard softening + survival protections
assert cfg.GUARD_TREND_REGIME_EXEMPTION and cfg.GUARD_EXEMPT_MIN_REGIME_CONF == 0.70
assert cfg.USE_EQUITY_TIERED_BREAKEVEN and cfg.BREAKEVEN_EQUITY_CUTOFF == 500.0
assert cfg.MAX_POSITIONS == 6

print("[PREFLIGHT] Config OK")
print(f"  setup:            {cfg.SETUP_TAG} v{cfg.SETUP_VERSION}")
print(f"  symbol / magic:   {cfg.SYMBOL} / {cfg.MAGIC_NUMBER}")
print(f"  entry gate:       prob>={cfg.META_THRESHOLDS['UNKNOWN']['buy']} "
      f"edge>={cfg.META_MIN_PROB_EDGE} regime_conf>={cfg.REGIME_MIN_CONFIDENCE}")
print(f"  queue:            cap={cfg.QUEUE_CAPACITY} score>={cfg.QUEUE_RELEASE_SCORE} "
      f"expiry={cfg.QUEUE_MAX_PENDING_MINUTES}min <=3/cycle <=2/side")
print(f"  campaign:         RISK {cfg.RISK_PCT_PER_TRADE}%/trade -> cap {cfg.CAMPAIGN_MAX_LOT} lots"
      f" | A+ x{cfg.A_PLUS_POSITION_COUNT} at prob>={cfg.A_PLUS_PROB_THRESHOLD}"
      f" once equity>=${cfg.A_PLUS_MIN_EQUITY:.0f}")
print(f"  protection:       BE@+1R below ${cfg.BREAKEVEN_EQUITY_CUTOFF:.0f} | "
      f"neg-time-stop {cfg.NEGATIVE_TIME_STOP_BARS} bars | max hold {cfg.MAX_HOLD_BARS}")
print(f"  daily limits:     {cfg.DAILY_LOSS_LIMIT:+.0f} / {cfg.DAILY_PROFIT_TARGET:+.0f} USD")

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
from regime_detector import RegimeRouter

stack = ModelStack().load(str(MODELS))
with open(MODELS / "feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)
assert len(feature_cols) == 76, f"expected 76 features, got {len(feature_cols)}"
smoke = stack.predict(pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols))
assert 0.0 <= smoke["buy_prob"] <= 1.0 and 0.0 <= smoke["sell_prob"] <= 1.0
print(f"  [OK] GBM smoke predict: buy={smoke['buy_prob']:.3f} sell={smoke['sell_prob']:.3f}")

router = RegimeRouter().load(str(MODELS))
assert set(router.hmm.state_map.values()) == \
    {"TREND_UP", "TREND_DOWN", "RANGING", "VOLATILE", "FLAT"}
print(f"  [OK] HMM regime router (retrained): {router.hmm.state_map}")

from signal_queue import SignalQueue
from entry_guards import hard_block_reason
from lot_campaign import compute_signal_lots
lot, n = compute_signal_lots(equity=1500, m15_atr=5.5, prob=0.80)
assert n == 2, "A+ dual entry not active above $1k in sizing math"
print(f"  [OK] Queue gate, hard-block guards, campaign sizing "
      f"(A+ sample: {n} x {lot} lots @ $1.5k)")

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
print("""
 M15 close -> 76 features -> GBM buy/sell probs
   -> HMM regime gate (conf>=0.50, trade_ok)
   -> meta gate (prob>=0.60, edge>=0.15, cooldown, position caps)
   -> SIGNAL QUEUE (never executed directly)

 M1 close  -> expire stale (>90min) -> leading-indicator score
   -> hard blocks: news / intraday extreme / H4 zone
      (extreme+zone skipped when HMM confidently agrees with side)
   -> release best-first (score>=4.0, <=3/cycle, <=2/side)
   -> risk-pct sizing (2%/trade; A+ = 2 tickets once equity>=$1k)
   -> market order, SL=1.0xATR / TP=2.0xATR

 Always     -> BE@+1R while equity<$500 | neg-time-stop 24 bars
            -> max hold 48 bars | daily halt at -250/+15000
            -> [STATUS] heartbeat every 5 min

 Expect few trades per day; queue depth and block reasons appear in
 the log. First A+ dual entries will only appear above $1k equity.
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
