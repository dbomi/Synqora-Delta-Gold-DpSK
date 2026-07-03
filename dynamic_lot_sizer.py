"""
Guarded dynamic lot sizing for the Delta system.

PATCHED: Runtime notebook/config LOT_SIZE is now treated as the requested base lot.

Key behavior:
- If USE_DYNAMIC_LOT_SIZING is False, return config.LOT_SIZE.
- If USE_DYNAMIC_LOT_SIZING is True, start from the current runtime config.LOT_SIZE
  rather than silently falling back to DYNAMIC_LOT_START_LOT.
- The old balance-step compounding rule can still increase size above the requested
  base as balance grows.
- Safety caps, broker volume limits, and rolling drawdown degrade still apply.
- Logs whenever the effective main lot changes, including the reason/cap path.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5
import config as runtime_config

logger = logging.getLogger("DynamicLotSizer")


@dataclass
class DynamicLotState:
    peak_equity: float = 0.0
    last_main_lot_log_key: str = ""
    last_branch_lot_log_key: str = ""

    def update_peak(self, equity: float) -> None:
        if equity and equity > self.peak_equity:
            self.peak_equity = float(equity)


_STATE = DynamicLotState()


def _cfg(name: str, default):
    """Read config values at call time so notebook overrides are respected."""
    return getattr(runtime_config, name, default)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_float(name: str, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except Exception:
        return float(default)


def _as_list(name: str, default):
    value = _cfg(name, default)
    return value if value is not None else default


def _requested_base_lot() -> float:
    """The notebook/config LOT_SIZE is the user's requested base lot."""
    return max(0.0, _as_float("LOT_SIZE", 0.02))


def _floor_to_step(value: float, step: float = 0.01) -> float:
    if step <= 0:
        step = 0.01
    return math.floor(float(value) / step + 1e-12) * step


def _symbol_volume_bounds(symbol: str) -> tuple[float, float, float]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.01, 1.0, 0.01
    min_vol = float(getattr(info, "volume_min", 0.01) or 0.01)
    max_vol = float(getattr(info, "volume_max", _as_float("DYNAMIC_LOT_MAX_LOT", 1.0)) or _as_float("DYNAMIC_LOT_MAX_LOT", 1.0))
    step = float(getattr(info, "volume_step", 0.01) or 0.01)
    return min_vol, max_vol, step


def _normalize_volume(symbol: str, lot: float) -> float:
    min_vol, broker_max, step = _symbol_volume_bounds(symbol)
    max_lot = _as_float("DYNAMIC_LOT_MAX_LOT", 1.0)
    lot = max(min_vol, min(float(lot), max_lot, broker_max))
    lot = _floor_to_step(lot, step)
    decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(max(min_vol, lot), decimals)


def _balance_step_lot(balance: float) -> float:
    """Original balance-step compounding path, kept for growth above the base lot."""
    first_balance = _as_float("DYNAMIC_LOT_FIRST_STEP_BALANCE", 1000.0)
    first_lot = _as_float("DYNAMIC_LOT_FIRST_STEP_LOT", 0.05)
    balance_step = max(_as_float("DYNAMIC_LOT_BALANCE_STEP", 500.0), 1.0)
    lot_step = _as_float("DYNAMIC_LOT_STEP", 0.05)
    max_lot = _as_float("DYNAMIC_LOT_MAX_LOT", 1.0)
    start_lot = _as_float("DYNAMIC_LOT_START_LOT", 0.02)

    balance = float(balance or 0.0)
    if balance < first_balance:
        return min(max_lot, start_lot)
    steps = math.floor((balance - first_balance) / balance_step)
    return min(max_lot, first_lot + lot_step * steps)


def raw_dynamic_lot(balance: float) -> float:
    """Return the uncapped lot before equity caps and drawdown degrade.

    Important: runtime config.LOT_SIZE is now the requested base. Dynamic sizing
    may increase above that base as the account grows, but it will not silently
    reduce to DYNAMIC_LOT_START_LOT. Later safety caps may still reduce it.
    """
    requested_base = _requested_base_lot()
    if not _as_bool(_cfg("USE_DYNAMIC_LOT_SIZING", True)):
        return requested_base

    # Keep the old compounding rule as an upside scaler, but never let it pull
    # the requested notebook/config base down to the old 0.02 start lot.
    dynamic_rule_lot = _balance_step_lot(balance)
    return min(_as_float("DYNAMIC_LOT_MAX_LOT", 1.0), max(requested_base, dynamic_rule_lot))


def capped_lot(balance: float) -> tuple[float, str]:
    lot = raw_dynamic_lot(balance)
    reason = "base_allowed"

    if not _as_bool(_cfg("USE_DYNAMIC_LOT_SIZING", True)):
        return lot, "fixed_config_lot"

    raw_lot = lot
    if balance < 2000.0:
        cap = _as_float("MAX_LOT_IF_EQUITY_BELOW_2000", 1.0)
        if lot > cap:
            lot = cap
            reason = f"equity_below_2000_cap:{raw_lot:.2f}->{lot:.2f}"
    elif balance < 5000.0:
        cap = _as_float("MAX_LOT_IF_EQUITY_BELOW_5000", 1.0)
        if lot > cap:
            lot = cap
            reason = f"equity_below_5000_cap:{raw_lot:.2f}->{lot:.2f}"

    return lot, reason


def drawdown_multiplier(equity: float, peak_equity: Optional[float] = None) -> float:
    if not _as_bool(_cfg("ROLLING_DRAWDOWN_DEGRADE", True)):
        return 1.0
    equity = float(equity or 0.0)
    peak = float(peak_equity if peak_equity is not None else _STATE.peak_equity or equity)
    if peak <= 0:
        return 1.0
    dd_pct = (equity - peak) / peak * 100.0
    mult = 1.0
    # Levels are negative thresholds, e.g. (-5, .75), (-10, .50), (-15, .25).
    for threshold, m in sorted(_as_list("DRAWDOWN_DEGRADE_LEVELS", []), key=lambda x: x[0], reverse=True):
        if dd_pct <= float(threshold):
            mult = float(m)
    return mult


def get_account_equity_balance() -> tuple[float, float]:
    info = mt5.account_info()
    if info is None:
        return 0.0, 0.0
    balance = float(getattr(info, "balance", 0.0) or 0.0)
    equity = float(getattr(info, "equity", balance) or balance)
    _STATE.update_peak(equity)
    return equity, balance


def _log_main_lot_once(symbol: str, requested_base: float, raw_lot: float, capped: float,
                       mult: float, effective: float, reason: str,
                       equity: float, balance: float) -> None:
    key = f"{symbol}|{requested_base:.2f}|{raw_lot:.2f}|{capped:.2f}|{mult:.3f}|{effective:.2f}|{reason}|{round(balance, 2)}"
    if key == _STATE.last_main_lot_log_key:
        return
    _STATE.last_main_lot_log_key = key
    logger.info(
        "[LOT SIZER] main requested_base=%.2f raw_dynamic=%.2f capped=%.2f "
        "drawdown_mult=%.2f effective_lot=%.2f reason=%s equity=%.2f balance=%.2f",
        requested_base, raw_lot, capped, mult, effective, reason, float(equity or 0.0), float(balance or 0.0),
    )


def get_main_lot_size(symbol: str, *, equity: Optional[float] = None, balance: Optional[float] = None) -> float:
    if equity is None or balance is None:
        equity, balance = get_account_equity_balance()

    requested_base = _requested_base_lot()
    if not balance or balance <= 0:
        effective = _normalize_volume(symbol, requested_base)
        _log_main_lot_once(symbol, requested_base, requested_base, requested_base, 1.0, effective, "no_balance_use_config_lot", float(equity or 0.0), float(balance or 0.0))
        return effective

    raw_lot = raw_dynamic_lot(balance)
    capped, reason = capped_lot(balance)
    mult = drawdown_multiplier(float(equity or balance))
    degraded = capped * mult
    if mult < 1.0:
        reason = f"{reason}+drawdown_degrade:{mult:.2f}"
    effective = _normalize_volume(symbol, degraded)
    _log_main_lot_once(symbol, requested_base, raw_lot, capped, mult, effective, reason, float(equity or balance), float(balance))
    return effective


def get_branch_lot_size(symbol: str, *, equity: Optional[float] = None, balance: Optional[float] = None) -> float:
    if equity is None or balance is None:
        equity, balance = get_account_equity_balance()

    mode = str(_cfg("BRANCH_LOT_MODE", "FIXED") or "FIXED").upper()
    branch_lot_size = _as_float("BRANCH_LOT_SIZE", 0.02)
    branch_lot_cap = _as_float("BRANCH_LOT_CAP", 0.10)

    if mode == "SAME_DYNAMIC":
        lot = get_main_lot_size(symbol, equity=equity, balance=balance)
        reason = "same_dynamic"
    elif mode == "HALF_DYNAMIC":
        lot = max(branch_lot_size, get_main_lot_size(symbol, equity=equity, balance=balance) * 0.50)
        reason = "half_dynamic_floor_branch_lot"
    elif mode == "CAP_0P10":
        lot = min(get_main_lot_size(symbol, equity=equity, balance=balance), branch_lot_cap)
        reason = "cap_0p10"
    else:
        lot = branch_lot_size
        reason = "fixed_branch_lot"

    effective = _normalize_volume(symbol, lot)
    key = f"{symbol}|{mode}|{effective:.2f}|{reason}|{round(float(balance or 0.0), 2)}"
    if key != _STATE.last_branch_lot_log_key:
        _STATE.last_branch_lot_log_key = key
        logger.info(
            "[LOT SIZER] branch mode=%s effective_lot=%.2f reason=%s equity=%.2f balance=%.2f",
            mode, effective, reason, float(equity or 0.0), float(balance or 0.0),
        )
    return effective


def reset_lot_peak(equity: float = 0.0) -> None:
    _STATE.peak_equity = float(equity or 0.0)
    _STATE.last_main_lot_log_key = ""
    _STATE.last_branch_lot_log_key = ""
