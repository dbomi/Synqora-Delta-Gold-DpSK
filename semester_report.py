"""
=============================================================================
SEMESTER REPORT — SYNQORA DELTA GOLD FABLE
Aggregates the six monthly replay ledgers (Jan-Jun 2026, each replayed with
the full tuned pipeline) and runs each month as an INDEPENDENT campaign
starting from $500 — testing behaviour across different market conditions.

All six months are out-of-sample for the GBM specialists (trained on data
through 2025-12-31).

Usage:  python semester_report.py
=============================================================================
"""

import glob
import numpy as np
import pandas as pd

from campaign_sim import run_campaign, stress_ledger, START_EQUITY

log = print

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
CAMPAIGNS = [
    ("FIXED 0.02",       ("FIXED", 0.02), 1),
    ("DELTA cap 5.0",    ("DELTA", 5.0),  1),
    ("RISK 2%",          ("RISK", 2.0),   1),
    ("RISK 2% + A+ x2",  ("RISK", 2.0),   2),
]


def load_month(m):
    df = pd.read_csv(f"logs/ledger_{m}.csv", parse_dates=["entry_time", "exit_time"])
    return df[df["outcome"] != "OPEN_END"].reset_index(drop=True)


def main():
    data = {m: load_month(m) for m in MONTHS}

    log("=" * 96)
    log("SEMESTER TEST — JAN-JUN 2026 | each month an independent campaign from "
        f"${START_EQUITY:.0f} | guards+exemption+tuned config")
    log("=" * 96)

    # ── Edge health per month (sizing-independent) ────────────────────────────
    log("\n--- MONTHLY EDGE HEALTH (fixed-lot view, sizing-independent) " + "-" * 33)
    log(f"{'month':>8} {'days':>5} {'trades':>7} {'win%':>6} {'sum R':>8} "
        f"{'avg R':>7} {'A+ n':>5} {'A+ win%':>8} {'worst day R':>12}")
    for m in MONTHS:
        df = data[m]
        move_r = df["pnl_usd"] / 2.0 / df["r_price"]
        ap = df[df["prob"] >= 0.70]
        ap_r = ap["pnl_usd"] > 0
        day_r = move_r.groupby(df["day"]).sum()
        log(f"{m:>8} {df['day'].nunique():>5} {len(df):>7} "
            f"{(df['pnl_usd'] > 0).mean() * 100:>5.1f}% {move_r.sum():>+8.1f} "
            f"{move_r.mean():>+7.2f} {len(ap):>5} {ap_r.mean() * 100:>7.1f}% "
            f"{day_r.min():>+12.1f}")

    # ── Campaign results per month ────────────────────────────────────────────
    for name, rule, mult in CAMPAIGNS:
        log(f"\n--- CAMPAIGN: {name} — each month from ${START_EQUITY:.0f} " + "-" * 40)
        log(f"{'month':>8} {'final equity':>13} {'return':>9} {'maxDD':>7} "
            f"{'worst day':>10} {'max lot':>8}")
        finals = []
        for m in MONTHS:
            r = run_campaign(data[m], rule, aplus_mult=mult)
            finals.append(r["final"])
            log(f"{m:>8} {r['final']:>12,.0f}{'*' if r['busted'] else ' '} "
                f"{r['ret_pct']:>+8.0f}% {r['max_dd_pct']:>6.1f}% "
                f"{r['worst_day']:>+10.0f} {r['max_lot']:>8.2f}")
        log(f"{'median':>8} {np.median(finals):>12,.0f}  | months profitable: "
            f"{sum(1 for f in finals if f > START_EQUITY)}/6 | worst month final: "
            f"{min(finals):,.0f}")

    # ── Stressed check on the WEAKEST month ───────────────────────────────────
    move_sums = {m: (data[m]["pnl_usd"] / 2.0 / data[m]["r_price"]).sum() for m in MONTHS}
    weakest = min(move_sums, key=move_sums.get)
    log(f"\n--- STRESSED MONTE CARLO ON WEAKEST MONTH ({weakest}: "
        f"{move_sums[weakest]:+.1f}R) " + "-" * 25)
    log("    (25% of winners flipped to -1R, +$0.20 slippage, 500 shuffled runs)")
    log(f"{'campaign':>18} {'median final':>13} {'p5 final':>10} {'p95 maxDD':>10} "
        f"{'P(loss)':>8} {'P(ruin)':>8}")
    rng = np.random.default_rng(11)
    for name, rule, mult in CAMPAIGNS:
        finals, dds, ruins = [], [], 0
        for _ in range(500):
            sdf = stress_ledger(data[weakest], rng)
            seq = rng.permutation(len(sdf))
            r = run_campaign(sdf, rule, aplus_mult=mult, sequence=seq)
            finals.append(r["final"])
            dds.append(r["max_dd_pct"])
            ruins += int(r["busted"])
        finals, dds = np.array(finals), np.array(dds)
        log(f"{name:>18} {np.median(finals):>12,.0f} {np.percentile(finals, 5):>10,.0f} "
            f"{np.percentile(dds, 95):>9.1f}% "
            f"{(finals < START_EQUITY).mean() * 100:>7.1f}% {ruins / 5:>7.1f}%")

    log("\nNotes: fills at M1 close incl. spread, no slippage (except stressed run);")
    log("GBMs trained through 2025-12-31 -> all six months out-of-sample for the models.")
    log("HMM was fitted on data through 2026-06 (regime model sees this period in-sample).")
    log("=" * 96)


if __name__ == "__main__":
    main()
