"""
=============================================================================
SCALE CAMPAIGN SIMULATOR — SYNQORA DELTA GOLD FABLE
Replays a replay_range trade ledger under different position-sizing
campaigns, compounding equity trade by trade. No MT5 needed.

Tested dimensions:
  - Fixed lots (baseline) vs Delta ladder vs risk-percent compounding
  - Lot growth up to 5.0
  - A+ dual entry (2 positions when signal prob >= threshold)
  - Margin cap, drawdown degrade, ruin detection
  - Monte Carlo: 1,000 shuffled trade orders per campaign to expose
    losing-streak risk the historical sequence may hide.

Usage:  python campaign_sim.py [ledger.csv]
=============================================================================
"""

import sys
import numpy as np
import pandas as pd

LEDGER = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/replay_range_2026-06-08_2026-07-03_guards.csv"

START_EQUITY   = 500.0
USD_PER_LOT    = 100.0        # $ per $1 gold move per 1.0 lot
MAX_LOT        = 5.0
MIN_LOT        = 0.01
LEVERAGE       = 500          # margin per lot ~= price*100/LEVERAGE
MARGIN_USE_CAP = 0.60         # total margin <= 60% of equity
RUIN_EQUITY    = 100.0
APLUS_PROB     = 0.70         # A+ signal: model prob >= this
DEGRADE        = [(-5.0, 0.75), (-10.0, 0.50), (-15.0, 0.25)]

log = print


def degrade_mult(equity, peak):
    if peak <= 0:
        return 1.0
    dd = (equity - peak) / peak * 100.0
    m = 1.0
    for thr, mult in sorted(DEGRADE, reverse=True):
        if dd <= thr:
            m = mult
    return m


def delta_ladder_lot(eq, cap):
    """Current Delta schedule, optionally uncapped above $5k."""
    if eq < 1000:
        lot = 0.02
    else:
        lot = 0.05 + 0.05 * int((eq - 1000) / 500)
    if eq < 2000:
        lot = min(lot, 0.10)
    elif eq < 5000:
        lot = min(lot, 0.30)
    return min(lot, cap)


def campaign_lot(rule, eq, peak, trade):
    """Base lot for one trade under a sizing rule (before A+ multiplier)."""
    kind, param = rule
    if kind == "FIXED":
        lot = param
    elif kind == "DELTA":
        lot = delta_ladder_lot(eq, cap=param)
    elif kind == "RISK":   # param = % of equity risked at 1R
        risk_usd = eq * param / 100.0
        lot = risk_usd / (trade["r_price"] * USD_PER_LOT)
    else:
        lot = 0.02
    lot *= degrade_mult(eq, peak)
    return max(MIN_LOT, min(round(lot, 2), MAX_LOT))


def stress_ledger(df, rng, flip_winner_prob=0.25, slippage_price=0.20):
    """
    Pessimistic copy of the ledger: a fraction of winners become full -1R
    losers (models decaying live: win rate ~63% -> ~47%) and every trade
    pays extra slippage. pnl_usd stays in 0.02-lot units.
    """
    out = df.copy()
    flip = (out["pnl_usd"] > 0) & (rng.random(len(out)) < flip_winner_prob)
    out.loc[flip, "pnl_usd"] = -(out.loc[flip, "r_price"] + 0.30) * 0.02 * USD_PER_LOT
    out["pnl_usd"] = out["pnl_usd"] - slippage_price * 0.02 * USD_PER_LOT
    return out


def run_campaign(df, rule, aplus_mult=1, sequence=None, track_curve=False,
                 aplus_min_equity=0.0):
    """
    Event-driven equity simulation over the ledger.
    sequence: optional row order (for Monte Carlo, run sequentially).
    Returns summary dict.
    """
    sequential = sequence is not None
    rows = df.iloc[sequence] if sequential else df

    eq, peak = START_EQUITY, START_EQUITY
    max_dd = 0.0
    day_pnl = {}
    max_conc, open_pos = 0, []
    max_lot_used = 0.0
    busted = False
    curve = []

    if sequential:
        events = [("X", i, r) for i, r in rows.iterrows()]   # entry+exit merged
    else:
        ev = []
        for i, r in rows.iterrows():
            ev.append((pd.Timestamp(r["entry_time"]), "E", i, r))
            ev.append((pd.Timestamp(r["exit_time"]), "C", i, r))
        ev.sort(key=lambda x: (x[0], 0 if x[1] == "C" else 1))
        events = ev

    pending = {}   # trade idx -> (lot, n_pos)

    def do_entry(i, r):
        nonlocal max_conc, max_lot_used
        base = campaign_lot(rule, eq, peak, r)
        n = aplus_mult if (r["prob"] >= APLUS_PROB and eq >= aplus_min_equity) else 1
        lot = base
        # margin cap on TOTAL open lots
        price = r["entry"]
        margin_per_lot = price * 100.0 / LEVERAGE
        open_lots = sum(l * k for l, k in pending.values())
        room = max(0.0, (eq * MARGIN_USE_CAP / margin_per_lot) - open_lots)
        total_want = min(lot * n, MAX_LOT)   # total exposure per signal capped
        total_lot = min(total_want, room)
        if total_lot < MIN_LOT:
            return
        lot_each = max(MIN_LOT, round(total_lot / n, 2))
        pending[i] = (lot_each, n)
        max_conc = max(max_conc, len(pending))
        max_lot_used = max(max_lot_used, lot_each * n)

    def do_exit(i, r):
        nonlocal eq, peak, max_dd, busted
        if i not in pending:
            return
        lot_each, n = pending.pop(i)
        move_price = r["pnl_usd"] / (0.02 * USD_PER_LOT)   # ledger ran 0.02 lots
        pnl = move_price * lot_each * n * USD_PER_LOT
        eq += pnl
        d = str(r["day"])
        day_pnl[d] = day_pnl.get(d, 0.0) + pnl
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100.0)
        if track_curve:
            curve.append(eq)
        if eq <= RUIN_EQUITY:
            busted = True

    for e in events:
        if busted:
            break
        if sequential:
            _, i, r = e
            do_entry(i, r)
            do_exit(i, r)
        else:
            _, typ, i, r = e
            if typ == "E":
                do_entry(i, r)
            else:
                do_exit(i, r)

    worst_day = min(day_pnl.values()) if day_pnl else 0.0
    return {
        "final": eq, "ret_pct": (eq / START_EQUITY - 1) * 100.0,
        "max_dd_pct": max_dd, "worst_day": worst_day,
        "max_conc": max_conc, "max_lot": max_lot_used,
        "busted": busted, "curve": curve,
    }


def main():
    df = pd.read_csv(LEDGER, parse_dates=["entry_time", "exit_time"])
    df = df[df["outcome"] != "OPEN_END"].reset_index(drop=True)
    log("=" * 88)
    log(f"SCALE CAMPAIGN SIMULATOR | ledger: {LEDGER} | {len(df)} trades | "
        f"start equity ${START_EQUITY:.0f} | leverage 1:{LEVERAGE}")
    log("=" * 88)

    # ── Signal-strength calibration: is "A+" actually better? ────────────────
    log("\n--- SIGNAL-STRENGTH CALIBRATION (release trades by model prob) " + "-" * 22)
    buckets = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]
    log(f"{'prob bucket':>12} {'n':>5} {'win%':>6} {'avg R':>7} {'sum R':>8}")
    for lo, hi in buckets:
        sub = df[(df["prob"] >= lo) & (df["prob"] < hi)]
        if len(sub) == 0:
            continue
        move_r = sub["pnl_usd"] / 2.0 / sub["r_price"]
        log(f"{f'{lo:.2f}-{hi:.2f}':>12} {len(sub):>5} "
            f"{(sub['pnl_usd'] > 0).mean() * 100:>5.1f}% {move_r.mean():>+7.2f} {move_r.sum():>+8.1f}")
    ap = df[df["prob"] >= APLUS_PROB]
    log(f"  A+ (prob>={APLUS_PROB}): {len(ap)}/{len(df)} trades "
        f"({len(ap) / len(df) * 100:.0f}%), win {(ap['pnl_usd'] > 0).mean() * 100:.1f}%, "
        f"avg {ap['pnl_usd'].mean() / 2 / ap['r_price'].mean():+.2f}R")

    # ── Concurrency in the real timeline ─────────────────────────────────────
    times = sorted([(r["entry_time"], 1) for _, r in df.iterrows()]
                   + [(r["exit_time"], -1) for _, r in df.iterrows()])
    conc, cmax = 0, 0
    for _, delta in times:
        conc += delta
        cmax = max(cmax, conc)
    log(f"\n  Max concurrent open positions in historical sequence: {cmax}")

    # ── Campaign matrix (historical order) ────────────────────────────────────
    campaigns = [
        ("FIXED 0.02 (baseline)",      ("FIXED", 0.02), 1),
        ("DELTA ladder cap 1.0",       ("DELTA", 1.0),  1),
        ("DELTA ladder cap 5.0",       ("DELTA", 5.0),  1),
        ("RISK 1%/trade cap 5.0",      ("RISK", 1.0),   1),
        ("RISK 2%/trade cap 5.0",      ("RISK", 2.0),   1),
        ("RISK 3%/trade cap 5.0",      ("RISK", 3.0),   1),
        ("RISK 1% + A+ x2",            ("RISK", 1.0),   2),
        ("RISK 2% + A+ x2",            ("RISK", 2.0),   2),
    ]

    log("\n--- CAMPAIGN RESULTS — HISTORICAL SEQUENCE (20 days) " + "-" * 32)
    log(f"{'campaign':>24} {'final equity':>13} {'return':>9} {'maxDD':>7} "
        f"{'worst day':>10} {'max lot':>8} {'conc':>5}")
    results = {}
    for name, rule, mult in campaigns:
        r = run_campaign(df, rule, aplus_mult=mult)
        results[name] = r
        log(f"{name:>24} {r['final']:>12,.0f}{'*' if r['busted'] else ' '} "
            f"{r['ret_pct']:>+8.0f}% {r['max_dd_pct']:>6.1f}% "
            f"{r['worst_day']:>+10.0f} {r['max_lot']:>8.2f} {r['max_conc']:>5}")
    log("  (* = account busted below $" + str(int(RUIN_EQUITY)) + ")")

    # ── Monte Carlo: shuffle trade order, 1000 runs per campaign ─────────────
    log("\n--- MONTE CARLO (1,000 shuffled orders per campaign, sequential) " + "-" * 20)
    log(f"{'campaign':>24} {'median final':>13} {'p5 final':>10} {'p95 maxDD':>10} "
        f"{'P(loss)':>8} {'P(DD>50%)':>10} {'P(ruin)':>8}")
    rng = np.random.default_rng(42)
    for name, rule, mult in campaigns:
        finals, dds, ruins = [], [], 0
        for _ in range(1000):
            seq = rng.permutation(len(df))
            r = run_campaign(df, rule, aplus_mult=mult, sequence=seq)
            finals.append(r["final"])
            dds.append(r["max_dd_pct"])
            ruins += int(r["busted"])
        finals, dds = np.array(finals), np.array(dds)
        log(f"{name:>24} {np.median(finals):>12,.0f} {np.percentile(finals, 5):>10,.0f} "
            f"{np.percentile(dds, 95):>9.1f}% "
            f"{(finals < START_EQUITY).mean() * 100:>7.1f}% "
            f"{(dds > 50).mean() * 100:>9.1f}% {ruins / 10:>7.1f}%")

    # ── STRESSED Monte Carlo: degraded edge + slippage ────────────────────────
    log("\n--- STRESSED MONTE CARLO (25% of winners flipped to -1R losers, " )
    log("    +$0.20 slippage/trade -> win rate ~47%; 500 runs per campaign) " + "-" * 18)
    log(f"{'campaign':>24} {'median final':>13} {'p5 final':>10} {'p95 maxDD':>10} "
        f"{'P(loss)':>8} {'P(DD>50%)':>10} {'P(ruin)':>8}")
    rng2 = np.random.default_rng(7)
    for name, rule, mult in campaigns:
        finals, dds, ruins = [], [], 0
        for _ in range(500):
            sdf = stress_ledger(df, rng2)
            seq = rng2.permutation(len(sdf))
            r = run_campaign(sdf, rule, aplus_mult=mult, sequence=seq)
            finals.append(r["final"])
            dds.append(r["max_dd_pct"])
            ruins += int(r["busted"])
        finals, dds = np.array(finals), np.array(dds)
        log(f"{name:>24} {np.median(finals):>12,.0f} {np.percentile(finals, 5):>10,.0f} "
            f"{np.percentile(dds, 95):>9.1f}% "
            f"{(finals < START_EQUITY).mean() * 100:>7.1f}% "
            f"{(dds > 50).mean() * 100:>9.1f}% {ruins / 5:>7.1f}%")

    log("\nNotes:")
    log("  - Monte Carlo runs trades sequentially (no overlap), so concurrency/margin")
    log("    effects are only in the historical-sequence table.")
    log("  - Ledger P&L includes spread cost, no slippage; fills at M1 close.")
    log("  - A+ x2 doubles position count on prob>=0.70 signals (two tickets).")
    log("=" * 88)


if __name__ == "__main__":
    main()
