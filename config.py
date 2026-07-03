"""
=============================================================================
SYNQORA DELTA GOLD FABLE - CONFIGURATION
=============================================================================
Clean rebuild of the Delta gold system around the three validated pillars:

  1. GBM BUY / SELL specialists (LightGBM + XGBoost ensemble).
     Walk-forward validated: PF=5.79, win_rate=72.2%, Sharpe=2.92 across
     5 out-of-sample splits (see Training results.txt in delta22).
  2. HMM regime intelligence (5-state Gaussian HMM + CUSUM), retrained
     fresh for this setup.
  3. Signal Queue Gate: model signals are queued, then released on M1
     leading-indicator confirmation instead of executing immediately.

LSTM, CatBoost meta-learner, and the delta22 live patch stack are
deliberately excluded — they were never part of the validated pipeline.

All system parameters live in this file.
=============================================================================
"""

SETUP_TAG = "Synqora Delta Gold Fable"
SETUP_VERSION = "1.1.0"

# ─────────────────────────────────────────────────────────────────────────────
# DEMO EXECUTION SAFETY
# ─────────────────────────────────────────────────────────────────────────────
# The live trader refuses to start on a real-money account while
# REQUIRE_DEMO_ACCOUNT is True. Trades ARE executed through MT5 (this is a
# demo validation run, not a dry run) — flip these only after demo sign-off.
REQUIRE_DEMO_ACCOUNT = True
ALLOW_REAL_ACCOUNT   = False
EXECUTE_TRADES       = True     # real MT5 orders on the (demo) account
HEARTBEAT_MINUTES    = 5        # [STATUS] heartbeat cadence in the log

# ─────────────────────────────────────────────────────────────────────────────
# INSTRUMENT & TIMEFRAMES
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL              = "GOLD"
PRIMARY_TF          = "M15"           # Training and signal timeframe
CONTEXT_TFS         = ["H1", "H4"]    # Higher-TF context features
FAST_TF             = "M1"            # Queue release / leading-indicator timeframe
QUEUE_ALIGN_TF      = "M5"            # M5 alignment check for queue scoring
LOT_SIZE            = 0.02            # Requested base lot (dynamic sizer may scale up)
MAGIC_NUMBER        = 880001
COMMENT             = "SynqoraFable"

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_START         = "2023-06-01"
TRAIN_END           = "2026-06-30"
VALIDATION_START    = "2026-04-01"
VALIDATION_END      = "2026-06-30"
MIN_BARS_REQUIRED   = 200

# ─────────────────────────────────────────────────────────────────────────────
# LABEL ENGINE
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: These match the geometry the shipped GBM models were trained and
# walk-forward validated on (Training results.txt: "TP=2.0×ATR, SL=1.0×ATR,
# max_bars=48"). If you change them you MUST retrain the GBM specialists.
TRIPLE_BARRIER_TP_ATR   = 2.0
TRIPLE_BARRIER_SL_ATR   = 1.0
TRIPLE_BARRIER_MAX_BARS = 48
MIN_R_RATIO             = 2.0
MFE_LOOKFORWARD         = 24

# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTOR (HMM + CUSUM) — retrained for Fable
# ─────────────────────────────────────────────────────────────────────────────
HMM_N_STATES            = 5
HMM_N_ITER              = 200
HMM_COVARIANCE_TYPE     = "full"
CUSUM_THRESHOLD         = 4.0
CUSUM_DRIFT             = 0.5
REGIME_WINDOW           = 60
REGIME_LABELS           = {
    0: "TREND_UP",
    1: "TREND_DOWN",
    2: "RANGING",
    3: "VOLATILE",
    4: "FLAT",
}
# Minimum HMM confidence for the regime gate to allow signal generation.
REGIME_MIN_CONFIDENCE   = 0.50

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
ATR_PERIOD              = 14
RSI_PERIOD              = 14
FAST_EMA                = 8
SLOW_EMA                = 21
TREND_EMA               = 50
VWAP_PERIOD             = 20
BOLLINGER_PERIOD        = 20
BOLLINGER_STD           = 2.0
MOMENTUM_PERIODS        = [3, 5, 10, 20]
VOLATILITY_PERIODS      = [5, 10, 20]
SEQUENCE_LEN            = 24          # kept for library compatibility (unused live)
FEATURE_EMBARGO_BARS    = 5

# ─────────────────────────────────────────────────────────────────────────────
# GBM MODEL PARAMS (as validated)
# ─────────────────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators":     1000,
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        6,
    "min_child_samples":50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "class_weight":     "balanced",
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}

XGB_PARAMS = {
    "n_estimators":     800,
    "learning_rate":    0.03,
    "max_depth":        5,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 1,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY GATE (meta-agent) — matches walk-forward validation exactly
# ─────────────────────────────────────────────────────────────────────────────
META_MIN_PROB_EDGE = 0.15
META_THRESHOLDS = {
    "TREND_UP":   {"buy": 0.60, "sell": 0.60},
    "TREND_DOWN": {"buy": 0.60, "sell": 0.60},
    "RANGING":    {"buy": 0.60, "sell": 0.60},
    "VOLATILE":   {"buy": 0.60, "sell": 0.60},
    "FLAT":       {"buy": 0.60, "sell": 0.60},
    "UNKNOWN":    {"buy": 0.60, "sell": 0.60},
}

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL QUEUE GATE
# ─────────────────────────────────────────────────────────────────────────────
# Model signals are never executed directly. They are queued and released
# only on M1 leading-indicator confirmation.

QUEUE_CAPACITY               = 20     # Max coexisting signals (BUY/SELL, all families)
QUEUE_MAX_PENDING_MINUTES    = 90     # Signal expiry — signals don't live forever
QUEUE_RELEASE_SCORE          = 4.0    # Min leading-indicator score to release
QUEUE_MAX_RELEASES_PER_CYCLE = 3      # Max signals released per M1 cycle
QUEUE_MAX_RELEASES_PER_SIDE  = 2      # Direction balance: max 2 BUY + 1 SELL or 1 BUY + 2 SELL
QUEUE_DEDUP_SAME_SIDE_BARS   = 1      # Skip enqueue if same side+family queued within N primary bars

# Leading-indicator score weights (M1 resolution, per queued signal)
QUEUE_SCORE_WEIGHTS = {
    "m1_momentum_zero_cross": 2.0,    # ROC turns +ve (BUY) / -ve (SELL) on last closed M1
    "tick_volume_spike":      1.5,    # M1 tick vol > 1.5× 20-bar avg AND close in direction
    "candle_body_proportion": 1.2,    # Last closed M1 body ≥ 60% of range, direction matches
    "pullback_to_queue_price":1.0,    # BUY: price dipped ≤ queue_price (SELL: ≥) — better entry
    "rejection_wick":         1.0,    # Lower wick ≥1.5× body (BUY) / upper wick (SELL)
    "roc_acceleration":       0.8,    # M1 ROC second derivative in signal direction
    "m5_alignment":           0.5,    # Last closed M5 candle aligns with signal side
    "spread_tightening":      0.3,    # Current spread ≤ 80% of queue-time spread
}

# Scoring parameters
QUEUE_M1_ROC_PERIOD          = 3      # M1 ROC lookback (bars)
QUEUE_VOL_SPIKE_MULT         = 1.5    # Tick volume spike multiple of rolling avg
QUEUE_VOL_AVG_BARS           = 20     # Rolling tick-volume average window (M1 bars)
QUEUE_BODY_MIN_PROPORTION    = 0.60   # Body ≥ 60% of range
QUEUE_WICK_BODY_RATIO        = 1.5    # Rejection wick ≥ 1.5× body
QUEUE_SPREAD_TIGHTEN_RATIO   = 0.80   # Spread ≤ 80% of entry-time spread

# ─────────────────────────────────────────────────────────────────────────────
# HARD-BLOCK GUARDS (block queue release regardless of score)
# ─────────────────────────────────────────────────────────────────────────────

# 1) News blackout.
# Static daily windows in UTC ("HH:MM"-"HH:MM"). Empty by default.
NEWS_BLACKOUT_ENABLED        = True
NEWS_STATIC_WINDOWS_UTC      = [
    # e.g. ("12:25", "12:45"),   # around typical US data drops
]
# Optional event calendar file (JSON list of {"time_utc": "2026-07-03T12:30:00",
# "name": "NFP"}). Blackout = [event - pre, event + post].
NEWS_EVENTS_FILE             = "news_events.json"
NEWS_PRE_EVENT_MINUTES       = 15
NEWS_POST_EVENT_MINUTES      = 15

# 2) Intraday extreme guard.
# Blocks release when entry would still chase the session extreme:
# BUY blocked within N×ATR(M15) of the session HIGH,
# SELL blocked within N×ATR(M15) of the session LOW.
# "Session" = current UTC trading day.
INTRADAY_EXTREME_GUARD_ENABLED = True
INTRADAY_EXTREME_ATR_MULT      = 1.5

# 3) H4 topzone guard.
# Blocks BUY when price sits in the top zone of the rolling H4 range
# (mirrored bottomzone applies to SELL).
H4_TOPZONE_GUARD_ENABLED     = True
H4_TOPZONE_LOOKBACK_BARS     = 20     # Rolling H4 range window
H4_TOPZONE_UPPER_PCT         = 0.85   # BUY blocked when pos-in-range ≥ 0.85
H4_BOTTOMZONE_GUARD_ENABLED  = True
H4_BOTTOMZONE_LOWER_PCT      = 0.15   # SELL blocked when pos-in-range ≤ 0.15

# Trend-regime exemption for the extreme/zone guards.
# 20-day replay evidence: in a persistent trend, the intraday-extreme and
# H4-zone guards systematically block the system's best trades (they gated
# out the sample's most profitable day entirely, -38% total). When the HMM
# regime confidently agrees with the signal direction (TREND_UP for BUY,
# TREND_DOWN for SELL), those two guards are skipped. The news blackout is
# NEVER exempted. Guards stay fully active in RANGING/VOLATILE/FLAT tape —
# rejection-at-extremes is exactly the counter-trend condition they exist for.
GUARD_TREND_REGIME_EXEMPTION = True
GUARD_EXEMPT_MIN_REGIME_CONF = 0.70   # HMM confidence needed for the exemption

# ─────────────────────────────────────────────────────────────────────────────
# SCALE CAMPAIGN (position sizing + A+ dual entry)
# ─────────────────────────────────────────────────────────────────────────────
# Campaign-sim evidence (297-trade ledger, incl. stressed Monte Carlo with win
# rate degraded to ~47% + slippage): risk-percent compounding beats the
# balance-step Delta ladder on both upside and stressed downside, and the
# signal-strength calibration justifies doubling strong signals
# (prob>=0.75 bucket: 73.8% win, +1.17R avg vs 46.3% win at 0.60-0.65).
LOT_SIZING_MODE          = "RISK_PCT"   # "RISK_PCT" | "DELTA" | "FIXED"
RISK_PCT_PER_TRADE       = 2.0          # % of equity risked at 1R per signal
CAMPAIGN_MAX_LOT         = 5.0          # cap per single position
MAX_TOTAL_SIGNAL_LOT     = 5.0          # cap on combined lots of one signal
CONTRACT_USD_PER_UNIT    = 100.0        # $ per $1 gold move per 1.0 lot (100 oz)
MARGIN_USE_CAP           = 0.60         # keep total margin <= 60% of free margin

# A+ dual entry: strong signals open two tickets (combined exposure still
# capped by MAX_TOTAL_SIGNAL_LOT). Two tickets rather than one bigger ticket
# so future management (e.g. one runner) can treat them independently.
A_PLUS_DUAL_ENTRY        = True
A_PLUS_PROB_THRESHOLD    = 0.70         # model prob for A+ classification
A_PLUS_POSITION_COUNT    = 2
A_PLUS_MIN_EQUITY        = 1000.0       # dual entry only once equity clears this.
# Semester test (Jan-Jun, each month from $500): the gate cut the stressed
# worst-case drawdown from 69.6% -> 47.0% and ruin risk 2.4% -> 0.8% while
# keeping ~85% of the upside — dual entry at min-lot on a tiny account was
# the dominant tail risk, not the risk percentage.

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
MAX_SPREAD_POINTS       = 80
DEVIATION               = 20
MAX_POSITIONS           = 6          # Max concurrent positions per direction
                                     # (raised from 3: A+ signals open 2 tickets;
                                     #  historical replay peaked at 6 concurrent)
COOLDOWN_BARS           = 3          # Min primary bars between same-direction entries
ORDER_RETRY_COUNT       = 3
ORDER_RETRY_DELAY       = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# POSITION MANAGER
# ─────────────────────────────────────────────────────────────────────────────
# Fable keeps management minimal and simulation-faithful: hard SL/TP set at
# entry from the label geometry, negative time stop, max-hold time stop.
MAX_HOLD_BARS           = 48         # Matches TRIPLE_BARRIER_MAX_BARS
USE_NEGATIVE_TIME_STOP  = True
NEGATIVE_TIME_STOP_BARS = 24         # Close positions still losing after N primary bars
DAILY_PROFIT_TARGET     = 15000.0    # USD — stop trading for the day
DAILY_LOSS_LIMIT        = -250.0     # USD — stop trading for the day

# Equity-tiered breakeven protection.
# 20-day replay evidence (2026-06-08..07-03): pure fixed TP/SL earns ~6% more
# than BE@1.0R overall, but BE@1.0R removes the "peaked +1R, died -1R"
# round-trips. So: survival phase (equity below the cutoff) arms breakeven at
# +1R; growth phase (equity above) runs the richer validated fixed geometry.
# Arming is per-position and one-way — SL is never loosened after arming.
USE_EQUITY_TIERED_BREAKEVEN = True
BREAKEVEN_EQUITY_CUTOFF     = 500.0  # BE active while account equity < this
BREAKEVEN_ARM_R             = 1.0    # Arm when profit reaches N × R (R = entry-to-SL)
BREAKEVEN_BUFFER_POINTS     = 5      # SL parked this many points beyond entry
                                     # (covers spread so a scratch closes ~$0)

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC LOT SIZING (guarded compounding, from validated Delta behaviour)
# ─────────────────────────────────────────────────────────────────────────────
USE_DYNAMIC_LOT_SIZING       = True
DYNAMIC_LOT_START_BALANCE    = 500.0
DYNAMIC_LOT_START_LOT        = 0.02
DYNAMIC_LOT_FIRST_STEP_BALANCE = 1000.0
DYNAMIC_LOT_FIRST_STEP_LOT   = 0.05
DYNAMIC_LOT_BALANCE_STEP     = 500.0
DYNAMIC_LOT_STEP             = 0.05
DYNAMIC_LOT_MAX_LOT          = 1.00
MAX_LOT_IF_EQUITY_BELOW_2000 = 0.10
MAX_LOT_IF_EQUITY_BELOW_5000 = 0.30
ROLLING_DRAWDOWN_DEGRADE     = True
DRAWDOWN_DEGRADE_LEVELS = [
    (-5.0, 0.75),
    (-10.0, 0.50),
    (-15.0, 0.25),
]

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
WF_N_SPLITS             = 5
WF_TEST_SIZE_BARS       = 2000
WF_EMBARGO_BARS         = FEATURE_EMBARGO_BARS
MIN_PROFIT_FACTOR       = 1.3
MIN_WIN_RATE            = 0.45
MAX_DRAWDOWN_PCT        = 0.15
MIN_SHARPE              = 0.8
STRESS_SPREAD_MULT      = 2.0
STRESS_SLIPPAGE_POINTS  = 20

# ─────────────────────────────────────────────────────────────────────────────
# SESSION WINDOWS (hour UTC)
# ─────────────────────────────────────────────────────────────────────────────
SESSIONS = {
    "SYDNEY":   (21, 6),
    "TOKYO":    (0, 9),
    "LONDON":   (7, 16),
    "NEW_YORK": (12, 21),
    "OVERLAP":  (12, 16),
}

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
import os
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR          = os.path.join(BASE_DIR, "models")
LOGS_DIR            = os.path.join(BASE_DIR, "logs")
DATA_CACHE_DIR      = os.path.join(BASE_DIR, "data_cache")

for _dir in [MODELS_DIR, LOGS_DIR, DATA_CACHE_DIR]:
    os.makedirs(_dir, exist_ok=True)
