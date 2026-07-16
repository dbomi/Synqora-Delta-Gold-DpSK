# Synqora Delta Gold Fable

> ML-driven gold (XAUUSD/GOLD) trading system for MetaTrader 5 — GBM signal ensembles,
> HMM regime intelligence, and a leading-indicator signal queue gate, with a
> risk-percent scale campaign validated over seven months of out-of-sample replay.

**Version:** 1.2.0 · **Python:** 3.11 · **Platform:** MetaTrader 5 (Windows)

> ⚠️ **Disclaimer** — This is research software for demo-account use. All results below
> are from simulated replays (M1-close fills, spread costs included, no slippage).
> They validate *relative* design decisions, not future returns. Nothing here is
> financial advice. Do not run on a live account without extensive demo validation.

---

## Table of contents

- [How it trades](#how-it-trades)
- [Validation results](#validation-results)
- [Signal queue gate](#signal-queue-gate)
- [Hard-block guards](#hard-block-guards)
- [M30 direction-context modulation](#m30-direction-context-modulation)
- [Position management](#position-management)
- [Scale campaign](#scale-campaign)
- [Starting equity: $500 vs $1,000](#starting-equity-500-vs-1000)
- [Quick start (demo)](#quick-start-demo)
- [Project structure](#project-structure)
- [Research tooling](#research-tooling)

---

## How it trades

```
M15 close ─► 76 features ─► GBM BUY/SELL probabilities (LightGBM + XGBoost)
   ─► HMM regime gate        (5-state HMM, confidence ≥ 0.50, CUSUM alarm)
   ─► meta gate              (prob ≥ 0.60, edge over opposite side ≥ 0.15,
                              cooldown 3 bars, position caps)
   ─► SIGNAL QUEUE           (signals are NEVER executed at generation time)

M1 close ─► expire stale (> 90 min) ─► leading-indicator scoring
   ─► hard blocks            (news / intraday extreme / H4 zone,
                              trend-regime exemption)
   ─► M30 direction filter   (1.15× lot on aligned trades, penalties 0)
   ─► release best-first     (score ≥ 4.0, ≤ 3/cycle, ≤ 2/side)
   ─► risk-percent sizing    (2%/trade; A+ signals open 2 tickets)
   ─► market order           SL = 1.0×ATR(M15), TP = 2.0×ATR(M15)

Open position ─► P17 ratchet (1.2R MFE → lock SL at +0.3R profit)
               ─► virtual lever if price gaps past intended SL
```

Three model components, all validated independently:

| Component | Role | Status |
|---|---|---|
| GBM BUY / SELL specialists | Signal probabilities | Walk-forward validated (see below) |
| 5-state Gaussian HMM + CUSUM | Regime gate + guard exemptions | Retrained on 72,612 M15 bars |
| Signal Queue Gate | M1 timing confirmation | New in Fable, replay-validated |

Deliberately **excluded** from the predecessor system: LSTM temporal model (never part
of the validated pipeline) and CatBoost meta-learner (trained on placeholder inputs).

## Validation results

**Walk-forward training validation** (5 out-of-sample splits, models trained on
2023-06 → 2025-12): PF **5.79**, win rate **72.2%**, Sharpe **2.92**, all splits PASS.

**Seven-month out-of-sample replay** (Jan–Jul 2026, models frozen at Dec 2025, full
pipeline incl. queue gate + guards, ~320–390 trades/month through Jun):

| Month | Trades | Win % | Edge (sum R) | A+ win % | Worst day |
|---|---|---|---|---|---|
| 2026-01 | 318 | 64.5% | +291R | 69.8% | −2.0R |
| 2026-02 | 342 | 59.1% | +256R | 61.5% | −2.4R |
| 2026-03 | 388 | 72.7% | +450R | 78.4% | +1.6R |
| 2026-04 | 353 | 59.2% | +270R | 67.2% | +1.0R |
| 2026-05 | 329 | 64.4% | +294R | 71.8% | +0.4R |
| 2026-06 | 353 | 62.0% | +293R | 69.0% | +0.0R |
| 2026-07 | 250 | 65.2% | +226R | 72.0% | +0.6R |

The edge held **every month**; strong signals (prob ≥ 0.75) win 73.8% at +1.17R average
vs 46.3% / +0.36R for marginal ones (0.60–0.65) — the basis for A+ dual entry.

## Signal queue gate

- Capacity **20**, FIFO eviction when full; BUY/SELL from all families coexist.
- Each slot stores `side, family, source_cid, queue_price, queue_time, m15_atr`
  (+ queue-time spread).
- Expiry after **90 minutes**; release requires leading-indicator **score ≥ 4.0**,
  up to **3 per M1 cycle, max 2 per side**, best score first.

| M1 leading indicator | Score |
|---|---|
| Momentum (ROC-3) zero-cross in signal direction | +2.0 |
| Tick volume > 1.5× 20-bar avg, close in direction | +1.5 |
| Candle body ≥ 60% of range, direction matches | +1.2 |
| Pullback to/beyond queue price | +1.0 |
| Rejection wick ≥ 1.5× body | +1.0 |
| ROC acceleration in direction | +0.8 |
| Last closed M5 candle aligned | +0.5 |
| Spread ≤ 80% of queue-time spread | +0.3 |

## Hard-block guards

Block release regardless of score (blocked signals stay queued and may release later):

1. **News blackout** — static UTC windows and/or a hot-reloaded `news_events.json`
   calendar (−15/+15 min around events). Never exempted.
2. **Intraday extreme** — BUY blocked within 1.5×ATR of the session high,
   SELL within 1.5×ATR of the session low.
3. **H4 topzone / bottomzone** — BUY blocked in the top 15% of the rolling
   20-bar H4 range (mirrored for SELL).

**Trend-regime exemption:** when the HMM reads TREND_UP with confidence ≥ 0.70,
BUY releases skip guards 2–3 (mirrored for TREND_DOWN/SELL). Measured impact over
20 days: +$4,175 → **+$5,086 (+22%)**, recovered the best day of the month that the
guards had previously gated to zero trades, while keeping full protection in
RANGING/VOLATILE/FLAT tape.

## M30 direction-context modulation

A lightweight filter that checks whether the intraday M30 candle direction (up/down)
aligns with the trade side at queue-release time. When aligned, the released trade
gets a **1.15× lot multiplier** — the base risk-percent lot is scaled up by 15%.
When opposed, no penalty (lot stays at base).

This was isolated as the winning variant after a three-way test:

| Variant | Config | P&L vs baseline |
|---|---|---|
| Lot-only (chosen) | `ALIGN_BOOST=0, LOT_ALIGN_BOOST=0.15` | **+$49.57** |
| Prob boost | `ALIGN_BOOST=0.03, LOT_ALIGN_BOOST=0` | inert (no signal crossed) |
| Fusion | `ALIGN_BOOST=0.03, LOT_ALIGN_BOOST=0.15` | inert on prob side, lot side active |

The prob boost was entirely inert because lowering the A+ threshold from 0.75 to 0.72
never captured an extra signal that wouldn't already have fired — the score-plus-prob
gate remains the binding constraint, not the raw prob threshold.

## Position management

Hard SL/TP geometry matching the training labels, with a surgical trailing ratchet
that addresses the specific loss DNA discovered in July 2026 analysis:

**Loss DNA finding (Jul 14–15):** 5 out of 6 losing trades went significantly
profitable (0.65R–1.86R MFE peak) before reversing to hit the −1R SL. The exits
were correct — the 1.0×ATR SL is the exact geometry the models were trained on —
but the sequence *profitably → loser* was the dominant damage pattern, not
"SL too wide" or "SL too tight."

- Hard SL/TP at entry: **SL = 1R (1.0×ATR), TP = 2R (2.0×ATR)** — the exact geometry
  the models were trained on.
- **P17 trailing ratchet:** when MFE reaches **1.2R**, SL is locked at **+0.3R profit**.
  If price gaps past the intended SL on the next tick, a **virtual lever** closes
  at market (eliminating gap-through risk). Tuned after testing showed a two-stage
  variant (second lock at +0.5R) was too aggressive and clipped 2R winners against
  deep pullbacks. Single-stage net effect over a 3-day sample: saved ~$53 from
  3 "green-before-dying" reversals, sacrificed ~$47 from 2 winners that pulled back
  past the 0.3R lock before reaching TP.
- Negative time stop (24 bars) and max-hold stop (48 bars).
- **Equity-tiered breakeven:** while equity < $500, positions reaching +1R get their
  SL moved to entry + buffer (survival insurance, ~−6% expectancy for zero
  "+1R peak → −1R loss" round-trips). Above $500: pure fixed geometry.
- Daily halt at −$250 loss / +$15,000 profit.

## Scale campaign

`LOT_SIZING_MODE = "RISK_PCT"`: each signal risks **2% of equity at 1R**, so lots
compound from 0.01–0.02 toward the **5.0-lot cap** as equity grows. Chosen over the
legacy balance-step ladder via a campaign matrix incl. stressed Monte Carlo (win rate
degraded to ~47% + slippage): risk-percent won on both upside and stressed downside —
its sizing shrinks automatically in drawdown, the ladder's does not.

**A+ dual entry:** signals with prob ≥ **0.75** open **two tickets** (combined ≤ 5.0 lots,
two tickets so one can later be managed as a runner) — but **only once equity ≥ $1,000**
(see below). Safety rails: drawdown degrade (−5/−10/−15% → 0.75/0.50/0.25× size),
margin headroom ≤ 60% of free margin, ≤ 6 positions per direction.

## Starting equity: $500 vs $1,000

Each month Jan–Jul was run as an **independent campaign** with the wired configuration
(RISK 2% + A+ ×2). Median monthly outcome by starting equity:

| | Start $500 | Start $1,000 |
|---|---|---|
| Median month final | ~$448k | ~$654k |
| Worst month final | $322k (Feb) | $364k (Feb) |
| Months profitable | 7/7 | 7/7 |
| Max monthly drawdown | 22.0% | 22.0% |
| **Stressed** p95 max drawdown | 47.0% | **37.6%** |
| **Stressed** ruin probability | 0.8% | **0.0%** |

*(Stressed = Monte Carlo on the weakest month with 25% of winners flipped to −1R
losers and extra slippage — a deliberately pessimistic edge.)*

**Why $1,000 is structurally better, not just "more money":**

1. **The min-lot floor stops binding.** At $500, the 0.01 minimum lot often risks
   *more* than the intended 2% — sizing can't shrink below the floor, so early losing
   streaks over-bite. This floor effect, not the risk percentage, was the dominant
   stressed tail risk found in testing.
2. **A+ dual entry is live from day one.** The `A_PLUS_MIN_EQUITY = $1,000` gate exists
   precisely because dual tickets at min-lot on a tiny account doubled the floor
   problem (gating it cut stressed worst-case drawdown 69.6% → 47.0% and ruin risk
   2.4% → 0.8% at the $500 start). Starting at $1,000 clears the gate immediately.
3. **Risk limits fit.** The −$250 daily loss halt is 25% of a $1,000 account versus
   50% of a $500 one.

The absolute dollar figures are frictionless-compounding artifacts — treat the
*relative* comparison as the finding: **fund at $1,000 if possible.**

## Quick start (demo)

Prerequisites: Windows, Python 3.11, MetaTrader 5 open and logged into a **demo**
account, Algo Trading enabled (toolbar button).

```bash
pip install -r requirements.txt

# Optional: refresh the regime intelligence (GBM models untouched)
python retrain_hmm.py

# Full retrain (data → labels → HMM → GBM → walk-forward → stress)
python trainer.py

# Demo-live runner (CLI)
python live_trader.py
```

**Recommended:** paste the whole of [`My NB Fable Runner.ipynb`](My%20NB%20Fable%20Runner.ipynb)
into a Jupyter notebook and run it. The cell:

1. Asserts the config matches the research-validated values (incl. M30 direction filter + P17 ratchet) — refuses to launch on drift.
2. Verifies all model artifacts load and smoke-predict (76 features).
3. Connects to MT5 with a **hard demo guard** (`trade_mode == DEMO`; a real account aborts).
4. Launches the runner with filtered live log streaming; interrupt the cell to stop.

Trades **are** executed on the demo account (`EXECUTE_TRADES = True`) — that is the
point of the demo phase. The runner logs a `[PREFLIGHT]` config banner at startup and
a `[STATUS]` heartbeat every 5 minutes. Session logs land in `logs/sessions/`.

Log tags: `[REGIME] [MODEL] [META] [QUEUE] [GUARD] [CAMPAIGN] [EXEC] [PM] [RATCHET] [STATUS]`.

## Project structure

| File | Role |
|---|---|
| `config.py` | All parameters, tagged and documented with the research evidence |
| `live_trader.py` | Demo-live orchestrator (closed-bar discipline, demo guard, heartbeat) |
| `run_demo_notebook.py` | Single-cell notebook launcher with full preflight |
| `signal_queue.py` | 20-slot FIFO queue + M1 leading-indicator release |
| `entry_guards.py` | News / intraday-extreme / H4-zone hard blocks + trend exemption |
| `direction_filter.py` | M30 candle-direction context for lot modulation |
| `lot_campaign.py` | Risk-percent sizing + A+ dual entry (pure math, unit-tested) |
| `model_stack.py` | GBM BUY/SELL specialists (LightGBM + XGBoost) |
| `regime_detector.py` | 5-state HMM + CUSUM regime router |
| `meta_agent.py` | Probability/edge/cooldown entry gate |
| `feature_engine.py` / `label_engine.py` / `data_engine.py` | Features, triple-barrier labels, MT5 data |
| `execution_engine.py` / `position_manager.py` | Orders + P17 trailing ratchet, time stops, tiered breakeven |
| `trainer.py` / `retrain_hmm.py` / `validation_engine.py` | Training + walk-forward validation |

## Research tooling

| Tool | Purpose |
|---|---|
| `replay_today.py` | Single-day pipeline replay with MFE + P17 ratchet what-ifs |
| `replay_range.py` | Multi-day / historical-window replay (`--start/--end/--out`) → trade ledger CSV |
| `campaign_sim.py` | Sizing-campaign matrix over a ledger, incl. stressed Monte Carlo |
| `semester_report.py` | Jan–Jul monthly campaign comparison |
| `test_queue_gate.py` / `test_breakeven.py` / `test_campaign_sizing.py` | MT5-free logic tests (44 checks) |

---

*Built on the validated core of the Delta gold system; rebuilt clean as
"Synqora Delta Gold Fable". MAGIC_NUMBER 880001 keeps Fable's positions isolated
from any legacy system on the same account.*
