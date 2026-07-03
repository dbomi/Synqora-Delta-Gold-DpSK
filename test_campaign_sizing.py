"""
Logic test for scale-campaign sizing (pure math, no MT5).
    python test_campaign_sizing.py
"""

from lot_campaign import compute_signal_lots

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, PASS if cond else FAIL))
    print(f"  [{PASS if cond else FAIL}] {name}")


ATR = 5.5   # 1R = 5.5 * $100 = $550 risk per 1.0 lot

print("-- risk-percent sizing --")
# $500 equity, 2% risk => $10 / $550 = 0.018 -> floors to 0.01
lot, n = compute_signal_lots(equity=500, m15_atr=ATR, prob=0.65)
check("small account sizes to min lot (0.01 x1)", (lot, n) == (0.01, 1))

# $10,000, 2% => $200 / $550 = 0.36
lot, n = compute_signal_lots(equity=10_000, m15_atr=ATR, prob=0.65)
check("$10k -> 0.36 lots x1", n == 1 and abs(lot - 0.36) < 1e-9)

# $200,000, 2% => $4,000 / $550 = 7.27 -> capped at 5.0
lot, n = compute_signal_lots(equity=200_000, m15_atr=ATR, prob=0.65)
check("cap at 5.0 lots", (lot, n) == (5.0, 1))

print("-- A+ dual entry --")
lot, n = compute_signal_lots(equity=10_000, m15_atr=ATR, prob=0.75)
check("A+ opens 2 tickets", n == 2)
check("A+ per-ticket lot = base (0.36 each, total 0.72)", abs(lot - 0.36) < 1e-9)

# A+ at huge equity: total capped at MAX_TOTAL_SIGNAL_LOT 5.0 -> 2.5 each
lot, n = compute_signal_lots(equity=200_000, m15_atr=ATR, prob=0.80)
check("A+ total capped at 5.0 (2.5 each)", n == 2 and abs(lot - 2.5) < 1e-9)

# A+ on tiny account: equity gate ($1k) suppresses dual entry -> single ticket
lot, n = compute_signal_lots(equity=500, m15_atr=ATR, prob=0.80)
check("below $1k equity A+ stays single (0.01 x1)", (lot, n) == (0.01, 1))
# Just above the gate: dual entry unlocks
lot, n = compute_signal_lots(equity=1500, m15_atr=ATR, prob=0.80)
check("above $1k equity A+ opens 2 tickets", n == 2)
# Unfundable second ticket falls back to 1 (margin room 0.015)
lot, n = compute_signal_lots(equity=1500, m15_atr=ATR, prob=0.80, margin_room_lots=0.015)
check("unfundable dual falls back to single 0.01", (lot, n) == (0.01, 1))

print("-- safety rails --")
lot, n = compute_signal_lots(equity=10_000, m15_atr=ATR, prob=0.65, dd_mult=0.50)
check("drawdown degrade halves size (0.18)", abs(lot - 0.18) < 1e-9)

lot, n = compute_signal_lots(equity=200_000, m15_atr=ATR, prob=0.65, margin_room_lots=1.2)
check("margin room caps lot (1.2)", abs(lot - 1.2) < 1e-9)

lot, n = compute_signal_lots(equity=0, m15_atr=ATR, prob=0.65)
check("zero equity -> no trade", (lot, n) == (0.0, 0))

lot, n = compute_signal_lots(equity=10_000, m15_atr=ATR, prob=0.65, margin_room_lots=0.0)
check("no margin room -> no trade", (lot, n) == (0.0, 0))

n_fail = sum(1 for _, r in results if r == FAIL)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed"
      + ("" if n_fail == 0 else f" — {n_fail} FAILED"))
raise SystemExit(1 if n_fail else 0)
