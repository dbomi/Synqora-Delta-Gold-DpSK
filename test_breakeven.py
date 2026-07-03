"""
Logic test for the equity-tiered breakeven (no MT5 needed — mt5 mocked).
    python test_breakeven.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import position_manager as pm

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, PASS if cond else FAIL))
    print(f"  [{PASS if cond else FAIL}] {name}")


def make_pos(side="BUY", entry=4000.0, r=5.0, price=None, sl=None):
    is_buy = side == "BUY"
    d = 1 if is_buy else -1
    return SimpleNamespace(
        ticket=111, magic=pm.MAGIC_NUMBER,
        type=0 if is_buy else 1,      # mt5.POSITION_TYPE_BUY == 0
        price_open=entry,
        price_current=price if price is not None else entry,
        sl=sl if sl is not None else entry - d * r,   # original SL at 1R
        tp=entry + d * 2 * r,                          # TP at 2R
        time=0, profit=0.0, volume=0.02,
    )


def run_case(pos, equity):
    """Returns the SL that would be set, or None if not armed."""
    calls = []
    fake_mt5 = SimpleNamespace(
        account_info=lambda: SimpleNamespace(equity=equity),
        symbol_info=lambda s: SimpleNamespace(point=0.01),
        POSITION_TYPE_BUY=0,
    )
    with patch.object(pm, "mt5", fake_mt5), \
         patch.object(pm, "modify_position_sl",
                      lambda ticket, new_sl, symbol: calls.append(new_sl) or True):
        pm._apply_equity_tiered_breakeven(pos, "GOLD")
    return calls[0] if calls else None


print("-- equity-tiered breakeven logic --")
# BUY, R=5 (entry 4000, SL 3995, TP 4010), equity 400 (< 500)
check("not armed below +1R (price +0.9R)",
      run_case(make_pos(price=4004.5), 400) is None)

sl = run_case(make_pos(price=4005.0), 400)
check("armed at +1.0R", sl is not None)
check("BE SL = entry + 5pt buffer (4000.05)", sl is not None and abs(sl - 4000.05) < 1e-9)

check("not armed when equity >= cutoff (600)",
      run_case(make_pos(price=4006.0), 600) is None)

check("one-way: no re-modify when SL already at/above BE",
      run_case(make_pos(price=4006.0, sl=4000.05), 400) is None)
check("re-arms if SL still original", run_case(make_pos(price=4006.0), 400) is not None)

sl = run_case(make_pos(side="SELL", price=3995.0), 400)   # SELL +1R
check("SELL armed at +1R with SL = entry - buffer (3999.95)",
      sl is not None and abs(sl - 3999.95) < 1e-9)
check("SELL not armed at +0.8R",
      run_case(make_pos(side="SELL", price=3996.0), 400) is None)

n_fail = sum(1 for _, r in results if r == FAIL)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed"
      + ("" if n_fail == 0 else f" — {n_fail} FAILED"))
raise SystemExit(1 if n_fail else 0)
