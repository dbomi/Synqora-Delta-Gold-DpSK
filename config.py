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
# REGIME DETECTOR — HMM (legacy) or Trend-based (ADX + multi-EMA)
# ─────────────────────────────────────────────────────────────────────────────
# Set REGIME_USE_TREND_DETECTOR = True to use the fast-adapting ADX/EMA-based
# trend detector instead of the 5-state HMM. The trend detector requires no
# training, adapts to current price action, and handles geopolitical shocks
# better because it reads live structure rather than historical patterns.
REGIME_USE_TREND_DETECTOR = True

# HMM + CUSUM (legacy, used when REGIME_USE_TREND_DETECTOR = False)
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

# Trend-based regime detector params (used when REGIME_USE_TREND_DETECTOR = True)
TREND_ADX_PERIOD          = 14      # ADX smoothing period
TREND_ADX_TREND_THRESH    = 22      # ADX >= this = trending
TREND_ADX_STRONG_THRESH   = 35      # ADX >= this = strong trend
TREND_EMA_FAST            = 8       # short-term (M15 bars ~ 2h)
TREND_EMA_MED             = 21      # medium-term (M15 bars ~ 5h)
TREND_EMA_SLOW            = 55      # long-term (M15 bars ~ 14h)
TREND_EMA_TREND           = 200     # major trend (M15 bars ~ 50h)
TREND_CONF_ADX_WEIGHT     = 0.5     # how much ADX contributes to confidence
TREND_CONF_ALIGN_WEIGHT   = 0.5     # how much multi-TF alignment contributes

# Minimum confidence for the regime gate to allow signal generation.
# Lowered from 0.50 → 0.25 (soft gate): lets more signals into the queue;
# the release gate + direction gate still filter bad entries at execution time.
REGIME_MIN_CONFIDENCE   = 0.25

# ── Option 2: Market-structure breakout override ──────────────────────────
# When price breaks 1h range with volume confirmation, override the regime
# gate so the GBM model can still generate signals during low-confidence
# regimes. The breakout confidence substitutes for regime confidence.
BREAKOUT_ENABLED           = True
BREAKOUT_LOOKBACK_BARS     = 60    # 1 hour of M1 bars for range calculation
BREAKOUT_VOLUME_MULT       = 1.3   # volume must exceed rolling avg by this multiple
BREAKOUT_MIN_STRENGTH      = 0.5   # minimum normalized breakout strength [0, 1]

# ── Option 3: Dual-path ADX+EMA signals ───────────────────────────────────
# When the GBM model gives low probabilities (both sides < 0.60) but the
# M15 ADX shows a strong trend (> 25) and EMAs are aligned directionally,
# generate a trend-following signal with a synthetic probability.
DUAL_PATH_ENABLED          = True
DUAL_PATH_ADX_MIN                = 20    # minimum ADX to generate a trend signal
DUAL_PATH_PROB_FALLBACK          = 0.55  # synthetic probability for ADX+EMA signals
DUAL_PATH_MAX_ATR_EXTENSION      = 2.0   # max ATR distance from 200 EMA; blocks buying after massive spikes

# ── Option 4: M30 Trend-Context Probability Modulator ─────────────────────
# Independent higher-TF direction filter using M30 data with momentum/ROC.
# Probability-modulation mode (no flipping): when the M30 trend disagrees
# with the M15 signal, the signal's effective probability is reduced (smaller
# position, stricter execution) rather than being flipped to the opposite
# direction. When the M30 trend agrees, the probability gets a modest boost.
#
# This preserves the GBM model's mean-reversion SELLs within an uptrend —
# the problem that destroyed the flip-mode filter — while reducing risk on
# counter-trend entries and rewarding trend-conforming entries.
#
# Modulation is proportional to M30 confidence (not binary):
#   aligned:   mod_prob = min(1.0, prob * (1 + m30_conf * ALIGN_BOOST))
#   opposed:   mod_prob = prob * (1 - m30_conf * OPPOSE_PENALTY)
#   neutral:   mod_prob = prob (unchanged)
#   low conf:  no modulation if m30_conf < MIN_CONFIDENCE
DIRECTION_FILTER_ENABLED                = True
DIRECTION_FILTER_ALIGN_BOOST           = 0.00    # boost prob when aligned (only boosts, never penalizes)
DIRECTION_FILTER_OPPOSE_PENALTY         = 0.00    # zero penalty — counter-trend signals untouched
DIRECTION_FILTER_MIN_CONFIDENCE         = 0.30    # minimum M30 confidence to modulate
# Direct lot-size modulation (separate from prob modulation)
DIRECTION_FILTER_LOT_ALIGN_BOOST       = 0.15
DIRECTION_FILTER_LOT_OPPOSE_PENALTY    = 0.00    # penalty multiplier when opposed
# M30 filter component weights (5-part composite)
DIRECTION_FILTER_EMA_WEIGHT            = 0.25
DIRECTION_FILTER_MOMENTUM_WEIGHT       = 0.30
DIRECTION_FILTER_ACCEL_WEIGHT          = 0.20
DIRECTION_FILTER_DI_WEIGHT             = 0.15
DIRECTION_FILTER_SWING_WEIGHT          = 0.10

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

# ── Regime-adaptive probability gate ────────────────────────────────────
# When regime confidence is low (< REGIME_ADAPTIVE_MIN_CONF), raise the
# GBM prob threshold by REGIME_ADAPTIVE_PROB_BOOST to filter unreliable
# signals in transitional/choppy markets.
REGIME_ADAPTIVE_MIN_CONF      = 0.50
REGIME_ADAPTIVE_PROB_BOOST    = 0.10

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

# Anti-chase release guard (added 2026-07-06 after live session analysis).
# A queued signal may only fill AT-OR-BETTER than its queue price, plus a
# small ATR tolerance. If price has already run in the signal direction the
# release is held; the signal stays queued (may fill on a retrace) or
# expires. Derivation: BTC loss-DNA (retrace entries beat chase entries) +
# live 2026-07-06 GOLD session (both losing clusters were chased fills).
ANTI_CHASE_ENABLED           = True
ANTI_CHASE_TOLERANCE_ATR     = 0.15   # xATR15 beyond queue price still acceptable

# Minimum queue age before a signal may release (proper delay: confirmation
# must come from M1 bars completed AFTER the signal was queued, not from the
# same impulse bar that generated it). 0 = old behaviour.
MIN_RELEASE_AGE_MINUTES      = 0   # A/B verdict 2026-07-07: age floor rejected
                                   # (age3: -$4.7k, age5: -$5.0k over 127 days,
                                   # no PF/DD gain — anti-chase already covers it)

# Wait-for-M1-close guard: prevents release when the signal was queued within
# the same M1 bar as the release cycle. This ensures M1 momentum indicators
# reflect a fully-closed bar, avoiding mid-bar entries where the forming bar
# has already reversed (e.g., live 2026-07-08 SELL entered at bounce bottom).
# Unlike MIN_RELEASE_AGE_MINUTES this is a 1-bar barrier, not a flat timer.
M1_WAIT_FOR_CLOSE = False

QUEUE_MAX_RELEASES_PER_CYCLE = 3      # Max signals released per M1 cycle
QUEUE_MAX_RELEASES_PER_SIDE  = 2      # Direction balance: max 2 BUY + 1 SELL or 1 BUY + 2 SELL
QUEUE_MAX_PER_SIDE            = 4      # Max same-side+family signals in queue; oldest replaced when exceeded

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

# M5 alignment precondition: if enabled, release requires M5 candle direction
# to match signal side, regardless of M1 score. This is a hard gate, not
# a score additive — it catches entries against the M5 trend.
M5_ALIGNMENT_PRECONDITION    = True    # Hard-gate: release only when last closed M5 candle direction matches signal side. Live 2026-07-10 validation: blocked 2 losers (-$50.04) vs 4 winners (+$12.52) = +$37.52 net improvement.

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
RISK_PCT_PER_TRADE       = 6.0          # % of equity risked at 1R per signal
CAMPAIGN_MAX_LOT         = 5.0          # cap per single position
MAX_TOTAL_SIGNAL_LOT     = 5.0          # cap on combined lots of one signal
CONTRACT_USD_PER_UNIT    = 100.0        # $ per $1 gold move per 1.0 lot (100 oz)
MARGIN_USE_CAP           = 0.60         # keep total margin <= 60% of free margin

# A+ dual entry: strong signals open two tickets (combined exposure still
# capped by MAX_TOTAL_SIGNAL_LOT). Two tickets rather than one bigger ticket
# so future management (e.g. one runner) can treat them independently.
A_PLUS_DUAL_ENTRY        = True
A_PLUS_PROB_THRESHOLD    = 0.75         # model prob for A+ classification; raised from 0.70 (81.4% WR bucket)
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

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCEMENTS P1–P10 (added 2026-07-08)
# ─────────────────────────────────────────────────────────────────────────────

# P1: Consecutive loss reduction — adaptive sizing after streaks
CONSECUTIVE_LOSS_REDUCTION     = True
CONSECUTIVE_LOSS_REDUCE_AFTER  = 2       # reduce after N consecutive losses
CONSECUTIVE_LOSS_REDUCE_BY     = 0.5     # multiply lot by this per level
CONSECUTIVE_LOSS_REDUCE_MAX    = 4       # max reduction levels
CONSECUTIVE_WIN_INCREASE_AFTER = 3       # increase after N consecutive wins
CONSECUTIVE_WIN_INCREASE_BY    = 1.25    # multiply lot by this

# P2: Regime-conditional queue release score threshold
QUEUE_RELEASE_SCORE_BY_REGIME = {
    "TREND_UP":   3.5,   # trend-aligned: easier release
    "TREND_DOWN": 3.5,
    "RANGING":    5.0,   # chop: need stronger confirmation
    "VOLATILE":   5.5,   # volatile: hardest to release
    "FLAT":       4.5,
    "UNKNOWN":    4.0,
}

# P3: Fresh-signal geometry — recalc SL/TP from current ATR at fill time
USE_FRESH_ATR_GEOMETRY = True

# P4: Synchronized release prevention — cluster gate
QUEUE_CLUSTER_GATE_ENABLED      = True
QUEUE_CLUSTER_GATE_WINDOW_MIN   = 15     # sliding window
QUEUE_CLUSTER_GATE_MAX_PER_SIDE = 2      # max per side in the window

# P5: Breakeven always on (overrides equity cutoff)
BREAKEVEN_ALWAYS_ON = True

# P13: Golden-hour scaling campaign — increase position size + dual entry
# during high-liquidity windows, reduce during dead zones.
# Evidence: golden hours (5-6, 16-18 UTC) have 92% win / 87% big-winner rate;
# dead hour (11 UTC) has 44% win rate. 184-trade replay: 2x golden adds +$1,551
# to raw P&L (+37%), 3x adds +$3,115 (+74%).
GOLDEN_HOUR_CAMPAIGN_ENABLED = True
# Enhanced: added hour 2 (Asia-London bridge, 88% WR, 6 big winners),
# raised golden multiplier to 2.5x, tightened dead-hour multiplier to 0.3x.
# Hour 9 added to dead hours (42% WR, negative P&L).
GOLDEN_HOURS_UTC             = [2, 5, 6, 16, 17, 18]
GOLDEN_HOUR_LOT_MULT         = 2.5        # multiply computed lot size
GOLDEN_HOUR_A_PLUS_THRESHOLD = 0.60       # lower threshold for dual entry
DEAD_HOURS_UTC               = [9, 11]
DEAD_HOUR_LOT_MULT           = 0.3        # reduce exposure in dead zones

# P6: Session-open block — block trades in the opening period of selected sessions
SESSION_OPEN_BLOCK_ENABLED      = True
SESSION_OPEN_BLOCK_MINUTES      = 45     # block first N minutes
SESSION_OPEN_BLOCK_SESSIONS     = ["SYDNEY", "TOKYO", "LONDON"]

# P7: Directional loss limiter — stop trading a side when it loses too much
DIRECTIONAL_LOSS_LIMIT_ENABLED = True
DIRECTIONAL_LOSS_LIMIT_PCT     = 0.50    # 50% of DAILY_LOSS_LIMIT per direction

# P8: Ensemble coherence — require buy_prob>sell_prob edge for BUY (and vice versa)
ENSEMBLE_COHERENCE_CHECK = True
MIN_COHERENCE_EDGE       = 0.10

# P9: Adaptive expiry — chased signals expire much faster
QUEUE_ADAPTIVE_EXPIRY_ENABLED      = True
QUEUE_MAX_PENDING_MINUTES_CHASED   = 15  # fast expire for chased signals

# P10: Margin validation — min free margin % after trade
MARGIN_VALIDATION_ENABLED = True
MARGIN_MIN_FREE_PCT      = 0.10

# P11: MFE-based trailing protection — auto-enabled in high volatility
# TUNED: ATR ratio raised from 1.3 to 1.8 to reduce false triggers.
# P14 does the same job better for most regimes; P11 only fires during
# extreme volatility spikes where an extra trail layer helps.
MFE_TRAIL_ENABLED        = True  # master switch
MFE_TRAIL_AUTO_MODE      = True   # auto-enable only during vol/regime conditions
MFE_ARM_BE_AT_R          = 1.0    # move SL to entry at MFE touch >= 1.0R
MFE_TRAIL_ACTIVATE_AT_R  = 1.5    # start trailing at MFE touch >= 1.5R
MFE_TRAIL_DISTANCE_R     = 0.75   # trail stop 0.75R behind max peak
MFE_TRAIL_BUFFER_POINTS  = 3      # extra buffer in points for trail stop
# Auto-trigger: enable P11 when ATR > N x median ATR over lookback
MFE_TRAIL_ATR_RATIO       = 1.8   # current ATR / median(ATR, 30) >= this
MFE_TRAIL_ATR_LOOKBACK    = 30    # bars for median ATR
MFE_TRAIL_REGIME_TRIGGERS = ["VOLATILE"]  # HMM regimes that also trigger P11

# P17: Trailing ratchet + virtual lever
# Converts "green-before-dying" losers to small wins by ratcheting
# SL to a small profit once the trade proves itself.
# Virtual lever closes at market if price gaps past the intended SL.
RATCHET_ENABLED         = True   # master switch
RATCHET_ARM_AT_R       = 1.2    # arm ratchet when MFE >= 1.2R
RATCHET_LOCK_AT_R      = 0.3    # ratchet SL to this many R in profit
RATCHET_VIRTUAL_LEVER  = True   # close at market if price gaps past intended SL

# ─────────────────────────────────────────────────────────────────────────────
# P12: Adaptive profit-target monitor (position monitor) — DEPRECATED
# Leaving config vars in place for reference, but the system uses P14 now.
# ─────────────────────────────────────────────────────────────────────────────
POSITION_MONITOR_ENABLED              = False
POSITION_MONITOR_SLEEP_SECONDS        = 5       # loop check interval
POSITION_MONITOR_M5_LOOKBACK          = 50      # bars for market classification
POSITION_MONITOR_VOLATILE_SIGMA       = 0.8     # pct change std threshold for VOLATILE
POSITION_MONITOR_TREND_MA_DIVERGENCE  = 0.20    # % MA5/MA20 divergence for TRENDING

# Profit targets per regime (total P/L)
POSITION_MONITOR_TARGET_VOLATILE      = 3.0     # $
POSITION_MONITOR_TARGET_RANGING       = 5.0     # $
POSITION_MONITOR_TARGET_TRENDING      = 12.0    # $

# Retrace thresholds per regime (fraction of max profit before close all)
POSITION_MONITOR_RETRACE_VOLATILE     = 0.70
POSITION_MONITOR_RETRACE_RANGING      = 0.80
POSITION_MONITOR_RETRACE_TRENDING     = 0.60

# Trend extension: how many consecutive increasing P/L bars before extending target
POSITION_MONITOR_TREND_EXTEND_BARS    = 3
POSITION_MONITOR_TREND_EXTEND_MULT    = 1.5

# Per-ticket multiplier for smaller vs larger positions
POSITION_MONITOR_TICKET_VOL_FACTOR    = 0.7     # volume > 0.05 gets this multiplier
POSITION_MONITOR_TICKET_MIN_TARGET    = 1.0     # per-ticket target never below this

# ─────────────────────────────────────────────────────────────────────────────
# P14: Regime-laddered profit protection (replaces P12)
# Once profit hits the first lock level, SL never goes below that threshold.
# Each regime has a ladder of profit locks + a trailing step behind peak.
# The core rule: once green, never red.
# Evidence: 94% of losers went green intra-trade (avg MFE 0.83R).
# This stops the round-trip by locking profit at each rung.
# ─────────────────────────────────────────────────────────────────────────────
REGIME_PROTECTION_ENABLED            = True

# Market classification (shared with what _classify_market uses)
REGIME_PROTECTION_M5_LOOKBACK        = 50
REGIME_PROTECTION_VOLATILE_SIGMA     = 0.8
REGIME_PROTECTION_MA_DIVERGENCE      = 0.20

# Each regime defines a list of (lock_R, trail_lookback_R) steps.
# At each step: when MFE reaches lock_R, SL is set to entry +/- (lock_R - trail_lookback_R) * R
# trail_lookback_R = 0 means SL goes exactly to entry (pure lock).
# trail_lookback_R > 0 means SL trails that far behind the lock level.
#
# Example for VOLATILE:
#   Step 1: at +0.15R, lock SL to entry (never go red). trail=0 means SL=entry.
#   Step 2: at +0.5R, lock SL to entry +0.2R (trail 0.3R behind lock).
#   Step 3: at +0.8R, lock SL to entry +0.5R (trail 0.3R behind).
#   Step 4: at +1.3R, lock SL to entry +1.0R (trail 0.3R behind).
# The steps are cumulative — once armed, SL never loosens.

REGIME_PROTECTION_STEPS = {
    # VOLATILE: first lock raised from 0.15R to 0.5R to avoid scratching
    # winners on noise. 0.15R fires on ~90% of trades (spread-level noise).
    # Rungs now wider apart with smaller trails to let runners breathe.
    "VOLATILE": [
        (0.5, 0.20),  # +0.5R -> SL=entry+0.3R (lock meaningful profit, not noise)
        (1.0, 0.35),  # +1.0R -> SL=entry+0.65R
        (1.6, 0.50),  # +1.6R -> SL=entry+1.1R
        (2.5, 0.70),  # +2.5R -> SL=entry+1.8R
    ],
    "TRENDING": [
        (0.5, 0.15),  # +0.5R -> SL=entry+0.35R (tight floor for trends)
        (1.2, 0.40),  # +1.2R -> SL=entry+0.8R
        (2.0, 0.60),  # +2.0R -> SL=entry+1.4R
        (3.5, 0.85),  # +3.5R -> SL=entry+2.65R
    ],
    "RANGING": [
        (0.15, 0.0),  # +0.15R -> SL=entry (kept tight — chop protection)
        (0.5, 0.20),  # +0.5R -> SL=entry+0.3R
        (0.9, 0.35),  # +0.9R -> SL=entry+0.55R
        (1.4, 0.50),  # +1.4R -> SL=entry+0.9R
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Exhaustion + divergence entry filter (BUY-only, M5 candle pattern guard)
# Blocks BUY entries when M5 shows a falling-knife pattern:
#   price has printed EXHAUSTION_MIN_CONSECUTIVE consecutive bearish M5
#   candles AND M5 ROC3 is still negative — buying into a dip that hasn't
#   reversed yet. SELL side was tested live and removed (kills winners).
# ─────────────────────────────────────────────────────────────────────────────
EXHAUSTION_FILTER_ENABLED       = True   # master switch
EXHAUSTION_MIN_CONSECUTIVE      = 3      # consecutive bearish M5 candles required
EXHAUSTION_ROC_BARS             = 3      # ROC lookback bars for divergence check

# ─────────────────────────────────────────────────────────────────────────────
# Pre-execution momentum alignment guard (blocks stale directional signals)
# Signals queued at M15 close wait up to 15min for the next M1 release.
# During that window, M1 momentum can reverse. This guard checks the
# last N M1 candles at release time — if price is moving strongly against
# the signal direction, the release is blocked.
# Trigger for the $17 BUY loss at 02:58 UTC: price was falling from 4072→4070
# during the 13min queue wait, but the anti-chase guard saw the lower price
# as a "pullback bargain" (price < queue_price for BUY = good). The M1
# momentum would have shown the drop was accelerating, not pulling back.
# ─────────────────────────────────────────────────────────────────────────────
MOMENTUM_ALIGNMENT_ENABLED              = True    # master switch
MOMENTUM_ALIGNMENT_LOOKBACK             = 3       # last N M1 bars for slope calc
MOMENTUM_ALIGNMENT_MIN_POINTS           = 0.3     # min avg points/bar to block (avoid noise)

# ─────────────────────────────────────────────────────────────────────────────
# Regime-direction gate (guards against rare counter-regime entries)
# Blocks SELL when HMM regime is TREND_UP at high confidence (conf>=0.80),
# and BUY when HMM regime is TREND_DOWN at high confidence.
# DNA analysis of the -$42.65 live loss today showed it was a SELL in a
# TREND_UP conf=1.00 environment — a scenario that had ZERO occurrences
# in the 15-day replay (blind spot).
# ─────────────────────────────────────────────────────────────────────────────
REGIME_DIRECTION_GATE_ENABLED        = True
REGIME_DIRECTION_GATE_CONFIDENCE     = 0.70   # min regime confidence to enforce (must be <= specific gate thresholds like BLOCK_BUY_DN_MIN_CONF)
REGIME_DIRECTION_GATE_BLOCK_SELL_UP  = True   # block SELL in TREND_UP
REGIME_DIRECTION_GATE_BLOCK_BUY_DN   = True   # block BUY in TREND_DOWN
# BUY-in-TREND_DOWN was disabled in the original HMM-based system because it
# killed 9 winners per 1 loser in replay. The new trend detector is more
# accurate, so we re-enable it but gate on a higher confidence threshold.
REGIME_DIRECTION_GATE_BLOCK_BUY_DN_MIN_CONF = 0.75  # only block when conf >= this

# Conflict-aware regime resolver. The HMM remains a protective gate, but its
# semantic label cannot hard-veto a strong opposite signal unless current M15,
# H1, M5 and M1 structure also support the HMM direction. When the HMM is
# contradicted by live structure, a high-quality opposite signal may enter at
# reduced risk rather than being discarded.
REGIME_CONFLICT_RESOLVER_ENABLED       = True
REGIME_CONFLICT_OVERRIDE_MIN_PROB      = 0.75
REGIME_CONFLICT_OVERRIDE_MIN_EDGE      = 0.50
REGIME_CONFLICT_OVERRIDE_MIN_SCORE     = 4.50
REGIME_CONFLICT_REDUCED_RISK_MULT      = 0.50
REGIME_CONFLICT_STRUCTURE_BARS         = 3
REGIME_CONFLICT_EMA_FAST               = 8
REGIME_CONFLICT_EMA_SLOW               = 21
REGIME_CONFLICT_MIN_CONTRADICTIONS     = 2
REGIME_CONFLICT_HARD_BLOCK_MIN_SUPPORT = 4

# ─────────────────────────────────────────────────────────────────────────────
# P15: Adaptive Peak Exit (independent profit maximizer)
# Three independent layers that detect different peak-failure modes, running
# on top of P14. P14 guarantees a trade never goes red; P15 catches the top.
#
# Layers (each fires independently):
#   1. Volatility trail — always trails a stop behind peak MFE at a fraction
#      of ATR. Catches slow grind-backs from the peak.
#   2. Momentum decay — monitors MFE growth rate over a sliding window.
#      When price stalls (rate drops near zero), tightens the trail to exit
#      quickly. Catches "loss of steam" before the reversal.
#   3. Volume climax — when M5 tick volume spikes 2× above average AND
#      the forming candle shows exhaustion (long wick, small body), exits
#      immediately. Catches blow-off tops.
#
# P15 does NOT modify broker SL. It evaluates independently and calls
# close_position() at market when conditions fire. P14's virtual stop
# remains the floor — P15 only exits if it detects a peak first.
# ─────────────────────────────────────────────────────────────────────────────
PEAK_EXIT_ENABLED                = True
# Layer 1: Volatility trail
#   Trail only activates once peak_fav >= PEAK_TRAIL_ATR_MULT * ATR (avoids
#   spread noise). trail distance = PEAK_TRAIL_ATR_MULT * ATR behind peak.
#   GOLD M5 ATR ≈ $8, spread ≈ $5.40 → 0.7×ATR=$5.60 > spread => clean.
PEAK_TRAIL_ATR_MULT              = 0.7    # trail distance as fraction of ATR
PEAK_TRAIL_ATR_LOOKBACK          = 14     # M5 bars for ATR calculation
# Layer 2: Momentum decay
PEAK_DECAY_ENABLED               = True
PEAK_DECAY_WINDOW_BARS           = 5      # consecutive stall checks before exit
PEAK_DECAY_MIN_MFE_R             = 1.0    # minimum MFE (in R) before decay fires
# Layer 3: Volume climax
PEAK_CLIMAX_ENABLED              = True
PEAK_CLIMAX_VOL_MULT             = 2.5    # volume spike vs 20-bar M5 average
PEAK_CLIMAX_WICK_RATIO           = 0.65   # wick / total range >= this = exhaustion
PEAK_CLIMAX_MIN_MFE_R            = 0.5    # minimum MFE before climax fires

# Buffer in points to avoid spread-induced scratch issues
REGIME_PROTECTION_BUFFER_POINTS      = 3

# ─────────────────────────────────────────────────────────────────────────────
# P16: Profit-retrace virtual close (catches sub-1R round-trips)
# Closes a position at market when profit retraces from a small peak back to
# near breakeven. Designed for sub-1R profits that P14's ladder step (0.15R)
# and broker min-stop-distance both miss.  No broker SL modify — pure virtual
# close via close_position().
# ─────────────────────────────────────────────────────────────────────────────
PROFIT_RETRACE_ENABLED           = True      # master switch
PROFIT_RETRACE_ARM_USD           = 2.0       # arm when peak profit >= $2
PROFIT_RETRACE_CLOSE_USD         = 0.8       # close when profit retraces to $0.80
