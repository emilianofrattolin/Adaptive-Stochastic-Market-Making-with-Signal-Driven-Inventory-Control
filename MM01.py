"""MM01 - adaptive market-making model.

This module implements a self-contained market-making strategy designed for
discrete limit-order-book simulations. The model combines stochastic-volatility
state estimation, order-flow pressure, spread-regime detection, and inventory
risk controls to decide when to quote passively, lean with short-term flow, or
liquidate exposure near the end of a session.

Model overview
--------------
MM01 maintains one state object per traded product. On every market update it
extracts the best bid/ask, mid price, spread, book imbalance, signed inventory,
and recent order-flow imbalance. These features are passed through a
Hull-White-style stochastic volatility filter with a Levy time-change overlay.
The filter produces a volatility regime and directional stress estimate that
drive quote width, quote placement, execution size, and inventory skew.

The quoting engine builds bid and ask candidates from four layers:

1. Fair-price layer:
   Uses the current mid price, micro-price imbalance, and order-flow imbalance
   to estimate a short-horizon fair value.
2. Volatility and spread layer:
   Expands or tightens the half-spread according to the filtered volatility
   regime, rolling spread quantiles, and a dynamic volatility ratio.
3. Inventory layer:
   Moves quotes away from inventory-increasing trades and improves the side
   that reduces inventory. When inventory approaches the configured limit, the
   model enters a danger zone with stronger skew, smaller close-side spreads,
   and more conservative opening sizes.
4. Execution-safety layer:
   Applies no-cross checks, best-edge rules, flow-aware joining logic, and
   liquidation behavior when time pressure becomes material.

Key features
------------
- Adaptive volatility filter: measurement noise is updated with an EWMA of
  squared innovations, bounded around the configured prior to avoid unstable
  jumps.
- Levy time-change overlay: large mid-price moves temporarily increase the
  effective volatility process and make quote placement more defensive.
- Dynamic order-flow imbalance: OFI decay adapts to short-horizon volatility,
  with caps that prevent directional signals from pushing quotes through the
  opposite best price.
- Rolling spread classifier: recent spread quantiles replace a static wide or
  tight threshold, allowing the model to adjust to changing book conditions.
- Inventory-aware execution: quote prices and sizes are asymmetric when the
  current position is large, especially close to the inventory limit.
- End-of-session liquidation: residual inventory is reduced through bounded
  chunks as the remaining session time falls below the configured threshold.
- Optional diagnostics: tagged orders, per-tick snapshots, counters, and
  counterfactual fields can be enabled without changing the order stream.

State persistence
-----------------
The rolling buffers and model states required for continuity are serialized
through traderData. This includes spread history, OFI peak history, volatility
references, adaptive measurement-noise estimates, and per-product filter state.
Cold-start defaults are conservative, so the model begins from configured
priors and only becomes more adaptive after enough market observations arrive.

Offline analysis utilities
--------------------------
The bottom of the file contains helper functions for evaluating fill logs
produced by a backtester:

- attribute_pnl_by_tag: PnL attribution by strategy tag and product.
- per_product_stats: per-product fill, notional, inventory, and PnL summary.
- compare_products: side-by-side product comparison.
- sensitivity_runner: one-parameter-at-a-time robustness checks.
- walk_forward_stats: rolling-window evaluation.
- block_bootstrap: confidence intervals for PnL and Sharpe-style metrics.
- session_split_eval: train/test style split evaluation.

The trading engine does not depend on the analysis helpers. They operate on
external lists, JSON files, or CSV fill logs, which keeps the live strategy path
compact and suitable for publication or reuse in other backtesting workflows.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
BASE_PARAMS: Dict[str, float] = {
    # UKF / HullWhiteSV core (invariato)
    "hw_kappa": 0.18,
    "hw_q": 0.25,
    "hw_R": 5.0,
    "hw_warmup": 30,
    "hw_z_hi": 0.95,
    "hw_z_lo": -0.92,

    # Levy time-change overlay (invariato)
    "hw_tc_kappa": 0.20,
    "hw_tc_theta": 1.0,
    "hw_tc_jump_alpha": 0.20,
    "hw_tc_jump_gate": 2.70,
    "hw_tc_jump_cap": 5.00,
    "hw_tc_q_mult": 1.35,
    "hw_tc_R_div": 0.75,
    "hw_tc_x_jump": 0.24,
    "hw_tc_tau_min": 0.75,
    "hw_tc_tau_max": 3.50,
    "hw_absret_alpha": 0.10,

    # Quote geometry
    "touch_offset": 1.0,
    "min_half": 1.0,
    "hi_widen": 0.9,
    "lo_tighten": 0.3,

    # BEST-EDGE QUOTING
    "wide_spread_tick": 10,
    "best_edge_offset": 0,

    "w_micro": 0.6,
    "w_imb": 0.55,
    "signal_cap": 1.6,

    # INVENTORY SKEW DIREZIONALE
    "inv_skew_price": 0.06,
    "inv_skew": 0.05,
    "inv_extra": 0.18,
    "stress_z_center": 0.55,
    "stress_z_k": 3.0,
    "stress_inv_center": 0.55,
    "stress_inv_k": 8.0,
    "inv_hard_frac": 0.65,
    "inv_shift_cap": 5.0,
    "asym_skew": True,
    "asym_skew_v2": True,

    "inv_widen": 1.8,

    "size": 6,
    "inv_limit": 16,
    "inv_soft": 10,
    "hi_size_mult": 0.6,
    "lo_size_mult": 1.0,

    "impact_coef": 0.05,
    "cross_cost": 1.0,
    "min_take_edge": 0.75,
    "take_enable": True,

    "liq_time_frac": 0.05,
    "liq_min_inv": 5,
    "liq_chunk": 4,

    "carry_penalty": 0.10,
    "liquidation_cost": 1.5,

    "session_len_default": 200_000,

    # OFI defaults
    "ofi_ema": 0.25,
    "w_ofi": 0.0,
    "ofi_cap": 1.5,
    "pepper_alpha_gate": 0.25,
    "pepper_one_sided_gate": 0.80,
    "pepper_join_tighten": 0.40,
    "pepper_lean_widen": 1.10,
    "pepper_impulse_gate": 1.10,
    "pepper_impulse_qty": 2,

    # STRATEGY25 selective imports from S24 (lighter defaults)
    "best_edge_flow_aware": True,
    "w_imb_flow": 0.55,
    "w_ofi_flow": 0.45,
    "flow_gate": 0.10,
    "cautious_offset": 1,
    "best_edge_neutral_mode": "cautious",

    "inv_exec_asym": True,
    "closing_half_shrink": 0.30,
    "closing_half_min_mult": 0.65,
    "closing_size_boost": 0.30,
    "opening_half_widen": 0.12,
    "opening_size_cut": 0.20,
    "opening_size_min_mult": 0.50,

    "skew_piecewise": True,
    "skew_deadband_frac": 0.10,
    "skew_kickin_frac": 0.60,
    "skew_kickin_mult": 0.75,
}

PARAM_OVERRIDES: Dict[str, Dict[str, float]] = {
    "ASH_COATED_OSMIUM": {
        "hw_z_hi": 0.96,
        "hw_z_lo": -0.92,
        "hw_tc_jump_gate": 2.50,
        "hw_tc_jump_cap": 4.25,
        "hw_tc_x_jump": 0.18,
        "hw_tc_q_mult": 1.25,
        "hw_tc_R_div": 0.70,

        "touch_offset": 1.0,
        "min_half": 1.0,
        "hi_widen": 0.7,
        "lo_tighten": 0.3,

        "wide_spread_tick": 8,
        "best_edge_offset": 0,

        "w_micro": 0.55,
        "w_imb": 0.35,
        "signal_cap": 1.5,

        "inv_skew_price": 0.07,
        "inv_extra": 0.14,
        "inv_hard_frac": 0.70,
        "inv_widen": 1.5,

        "size": 7,
        "inv_limit": 18,
        "inv_soft": 12,
        "hi_size_mult": 0.75,

        "take_enable": True,
        "min_take_edge": 0.70,

        "liq_time_frac": 0.04,
        "liq_chunk": 5,
    },
    "INTARIAN_PEPPER_ROOT": {
        "hw_z_hi": 0.97,
        "hw_z_lo": -0.93,

        "touch_offset": 1.2,
        "min_half": 1.0,
        "hi_widen": 1.1,
        "lo_tighten": 0.3,

        "wide_spread_tick": 8,
        "best_edge_offset": 0,

        "w_micro": 0.35,
        "w_imb": 0.85,
        "signal_cap": 1.3,

        "inv_skew_price": 0.08,
        "inv_extra": 0.30,
        "inv_hard_frac": 0.55,
        "stress_z_center": 0.45,
        "stress_inv_center": 0.45,
        "inv_widen": 2.2,

        "size": 4,
        "inv_limit": 14,
        "inv_soft": 9,
        "hi_size_mult": 0.5,

        "take_enable": True,
        "min_take_edge": 1.50,

        "liq_time_frac": 0.06,
        "liq_min_inv": 3,
        "liq_chunk": 3,

        "ofi_ema": 0.25,
        "w_ofi": 1.15,
        "ofi_cap": 1.80,

        "pepper_alpha_gate": 0.25,
        "pepper_one_sided_gate": 0.80,
        "pepper_join_tighten": 0.40,
        "pepper_lean_widen": 0.50,

        "pepper_impulse_gate": 1.10,
        "pepper_impulse_qty": 2,

        "hw_tc_kappa": 0.24,
        "hw_tc_jump_alpha": 0.24,
        "hw_tc_jump_gate": 2.90,
        "hw_tc_jump_cap": 5.80,
        "hw_tc_q_mult": 1.45,
        "hw_tc_R_div": 0.80,
        "hw_tc_x_jump": 0.30,
        "hw_tc_tau_max": 4.00,
    },
}

DEFAULT_UNKNOWN_PARAMS: Dict[str, Any] = {
    "wide_spread_tick": 999_999,
    "best_edge_offset": 1,
    "min_half": 2.0,
    "touch_offset": 0.5,
    "size": 1,
    "inv_limit": 4,
    "inv_soft": 2,
    "take_enable": False,
    "asym_skew_v2": True,
    "inv_skew_price": 0.50,
    "inv_widen": 3.0,
    "liq_time_frac": 0.10,
    "liq_min_inv": 1,
    "liq_chunk": 1,
}


def params_for(product: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ritorna i parametri per un prodotto. `overrides` (opzionale) e' un dict
    applicato SOPRA la merge base+product, ed e' usato dal sensitivity runner
    per perturbare singoli valori senza toccare BASE_PARAMS/PARAM_OVERRIDES.
    """
    p = dict(BASE_PARAMS)
    if product in PARAM_OVERRIDES:
        p.update(PARAM_OVERRIDES[product])
    else:
        p.update(DEFAULT_UNKNOWN_PARAMS)
    if overrides:
        p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# HullWhiteSV (invariato vs S20)
# ---------------------------------------------------------------------------
class HullWhiteSV:
    def __init__(
        self,
        kappa, q, R,
        z_hi=0.8, z_lo=-0.8, warmup=30,
        tc_kappa=0.18, tc_theta=1.0,
        tc_jump_alpha=0.18, tc_jump_gate=2.2, tc_jump_cap=4.0,
        tc_q_mult=1.25, tc_R_div=0.70, tc_x_jump=0.22,
        tc_tau_min=0.75, tc_tau_max=3.50,
        absret_alpha=0.08,
    ):
        self.kappa = float(kappa); self.q = float(q); self.R = float(R)
        self.z_hi = float(z_hi); self.z_lo = float(z_lo); self.warmup = int(warmup)
        self.tc_kappa = float(tc_kappa); self.tc_theta = float(tc_theta)
        self.tc_jump_alpha = float(tc_jump_alpha); self.tc_jump_gate = float(tc_jump_gate)
        self.tc_jump_cap = float(tc_jump_cap); self.tc_q_mult = float(tc_q_mult)
        self.tc_R_div = float(tc_R_div); self.tc_x_jump = float(tc_x_jump)
        self.tc_tau_min = float(tc_tau_min); self.tc_tau_max = float(tc_tau_max)
        self.absret_alpha = float(absret_alpha)

        self.x: Optional[float] = None
        self.P: float = 1.0
        self.mu: Optional[float] = None
        self.last_mid: Optional[float] = None
        self.n: int = 0
        self.x_mean: float = 0.0
        self.x_M2: float = 0.0
        self.tau: float = 1.0
        self.jump_ema: float = 0.0
        self.last_jump_score: float = 0.0
        self.ewa_abs_r: float = 1.0
        # MODIFIED [4 — adaptive hw_R]: initialize EWMA of squared innovations
        # at the fixed R so early-session behavior is identical to S20 until
        # innovation data accumulates.
        self.R_estimated: float = float(R)

    @staticmethod
    def _clamp_static(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def update(self, mid: float) -> Tuple[float, float, str]:
        if self.last_mid is None:
            self.last_mid = float(mid)
            return 1.0, 0.0, "normal"

        r = float(mid) - self.last_mid
        self.last_mid = float(mid)
        abs_r = abs(r)
        self.ewa_abs_r = (1.0 - self.absret_alpha) * self.ewa_abs_r + self.absret_alpha * max(1e-6, abs_r)

        scale = max(0.25, self.ewa_abs_r)
        shock = abs_r / scale
        jump_score = max(0.0, shock - self.tc_jump_gate)
        jump_score = min(jump_score, self.tc_jump_cap)
        self.last_jump_score = jump_score
        self.jump_ema = (1.0 - self.tc_jump_alpha) * self.jump_ema + self.tc_jump_alpha * jump_score

        tau = self.tau + self.tc_kappa * (self.tc_theta - self.tau) + self.tc_jump_alpha * self.jump_ema
        self.tau = self._clamp_static(tau, self.tc_tau_min, self.tc_tau_max)

        y = math.log(r * r + 1e-6)

        if self.x is None:
            self.x = y
            self.mu = y

        q_eff = self.q * (1.0 + self.tc_q_mult * max(0.0, self.tau - 1.0))

        # MODIFIED [4 — adaptive hw_R]: blend fixed prior R with EWMA of squared
        # innovations. 70% prior / 30% estimated; clamped to [0.5x, 2x] of R.
        # The innovation is (y - x_pred), computed below before applying the
        # gain. We use the PREVIOUS step's R_estimated in the current R_adaptive
        # (the update to R_estimated happens after innovation is observed).
        R_adaptive = 0.70 * self.R + 0.30 * self.R_estimated
        R_adaptive = max(self.R * 0.5, min(self.R * 2.0, R_adaptive))
        R_eff = R_adaptive / (1.0 + self.tc_R_div * max(0.0, self.tau - 1.0))
        R_eff = max(1e-6, R_eff)

        x_pred = self.x + self.kappa * (self.mu - self.x)
        x_pred += self.tc_x_jump * self.jump_ema

        # MODIFIED [4]: innovation = observed - predicted (in log-r^2 space).
        innovation = y - x_pred
        hw_R_ewma_alpha = 0.05  # slow adaptation per spec
        self.R_estimated = hw_R_ewma_alpha * (innovation * innovation) \
                           + (1.0 - hw_R_ewma_alpha) * self.R_estimated

        P_pred = self.P + q_eff
        K = P_pred / (P_pred + R_eff)
        self.x = x_pred + K * innovation
        self.P = (1.0 - K) * P_pred
        self.P = max(1e-6, self.P)
        self.mu += 0.002 * (self.x - self.mu)

        self.n += 1
        d = self.x - self.x_mean
        self.x_mean += d / self.n
        self.x_M2 += d * (self.x - self.x_mean)
        x_std = math.sqrt(self.x_M2 / max(1, self.n - 1)) if self.n > 1 else 1.0
        z = (self.x - self.x_mean) / max(0.5, x_std)

        if self.n < self.warmup:
            regime = "normal"
        elif z > self.z_hi:
            regime = "high"
        elif z < self.z_lo:
            regime = "low"
        else:
            regime = "normal"

        sigma = math.exp(0.5 * self.x)
        return sigma, z, regime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kappa": self.kappa, "q": self.q, "R": self.R,
            "z_hi": self.z_hi, "z_lo": self.z_lo, "warmup": self.warmup,
            "tc_kappa": self.tc_kappa, "tc_theta": self.tc_theta,
            "tc_jump_alpha": self.tc_jump_alpha, "tc_jump_gate": self.tc_jump_gate,
            "tc_jump_cap": self.tc_jump_cap, "tc_q_mult": self.tc_q_mult,
            "tc_R_div": self.tc_R_div, "tc_x_jump": self.tc_x_jump,
            "tc_tau_min": self.tc_tau_min, "tc_tau_max": self.tc_tau_max,
            "absret_alpha": self.absret_alpha,
            "x": self.x, "P": self.P, "mu": self.mu, "last_mid": self.last_mid,
            "n": self.n, "x_mean": self.x_mean, "x_M2": self.x_M2,
            "tau": self.tau, "jump_ema": self.jump_ema,
            "last_jump_score": self.last_jump_score, "ewa_abs_r": self.ewa_abs_r,
            # MODIFIED [4]: persist adaptive R estimator.
            "R_estimated": self.R_estimated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HullWhiteSV":
        obj = cls(
            d["kappa"], d["q"], d["R"],
            d.get("z_hi", 0.8), d.get("z_lo", -0.8), d.get("warmup", 30),
            d.get("tc_kappa", 0.18), d.get("tc_theta", 1.0),
            d.get("tc_jump_alpha", 0.18), d.get("tc_jump_gate", 2.2),
            d.get("tc_jump_cap", 4.0), d.get("tc_q_mult", 1.25),
            d.get("tc_R_div", 0.70), d.get("tc_x_jump", 0.22),
            d.get("tc_tau_min", 0.75), d.get("tc_tau_max", 3.50),
            d.get("absret_alpha", 0.08),
        )
        obj.x = d.get("x"); obj.P = float(d.get("P", 1.0)); obj.mu = d.get("mu")
        obj.last_mid = d.get("last_mid"); obj.n = int(d.get("n", 0))
        obj.x_mean = float(d.get("x_mean", 0.0)); obj.x_M2 = float(d.get("x_M2", 0.0))
        obj.tau = float(d.get("tau", 1.0)); obj.jump_ema = float(d.get("jump_ema", 0.0))
        obj.last_jump_score = float(d.get("last_jump_score", 0.0))
        obj.ewa_abs_r = float(d.get("ewa_abs_r", 1.0))
        # MODIFIED [4]: restore adaptive R estimator; default to the fixed R
        # so a missing key produces identical-to-S20 warm start.
        obj.R_estimated = float(d.get("R_estimated", obj.R))
        return obj


class OFIState:
    # MODIFIED [1 — OFI decay]: rolling window for fast-reset peak tracking.
    # Hardcoded (no new param in BASE_PARAMS): reset trigger when |alpha| < 10% of
    # rolling max over last 20 updates; on trigger multiply alpha by 0.5.
    _PEAK_WINDOW: int = 20
    _RESET_THRESHOLD: float = 0.10
    _RESET_FACTOR: float = 0.5

    def __init__(self, ema: float):
        self.ema = float(ema)
        self.prev_bid: Optional[float] = None
        self.prev_ask: Optional[float] = None
        self.prev_bv: float = 0.0
        self.prev_av: float = 0.0
        self.alpha: float = 0.0
        # MODIFIED [1]: rolling buffer of |alpha| for fast-reset mean-reversion.
        self.peak_buf: List[float] = []

    def update(self, bid: float, bv: float, ask: float, av: float,
               vol_ratio: float = 1.0) -> float:
        """Update OFI EMA. `vol_ratio` (>=0) is short-horizon realized vol
        divided by baseline; pass 1.0 to disable dynamic decay (default)."""
        bid = float(bid); ask = float(ask)
        bv = max(1.0, float(bv)); av = max(1.0, float(av))

        if self.prev_bid is None or self.prev_ask is None:
            self.prev_bid, self.prev_ask = bid, ask
            self.prev_bv, self.prev_av = bv, av
            return 0.0

        e = (
            (1.0 if bid >= self.prev_bid else 0.0) * bv
            - (1.0 if bid <= self.prev_bid else 0.0) * self.prev_bv
            - (1.0 if ask <= self.prev_ask else 0.0) * av
            + (1.0 if ask >= self.prev_ask else 0.0) * self.prev_av
        )
        depth = max(1.0, bv + av)
        raw = e / depth

        # BEFORE: self.alpha = (1.0 - self.ema) * self.alpha + self.ema * raw
        # MODIFIED [1 — dynamic OFI decay]:
        # Code convention: self.ema is weight on NEW raw observation.
        # User convention: alpha_user = weight on OLD value = 1 - self.ema.
        # At high vol, accelerate decay (shrink alpha_user); never slower than base,
        # never faster than 40% of base.
        alpha_user_base = 1.0 - self.ema
        vr = max(0.0, min(float(vol_ratio), 3.0))  # cap at 3x per spec
        alpha_user_dyn = alpha_user_base * (1.0 - 0.5 * (vr - 1.0) / 2.0)
        alpha_user_dyn = max(alpha_user_dyn, alpha_user_base * 0.4)  # 40% floor
        alpha_user_dyn = min(alpha_user_dyn, alpha_user_base)        # <= base
        self.alpha = alpha_user_dyn * self.alpha + (1.0 - alpha_user_dyn) * raw

        # MODIFIED [1 — OFI fast-reset]:
        # If |alpha| drops below 10% of rolling peak over last 20 ticks,
        # halve the current EMA to accelerate mean-reversion.
        abs_alpha = abs(self.alpha)
        self.peak_buf.append(abs_alpha)
        if len(self.peak_buf) > self._PEAK_WINDOW:
            self.peak_buf.pop(0)
        if len(self.peak_buf) >= 5:
            recent_peak = max(self.peak_buf)
            if recent_peak > 1e-9 and abs_alpha < self._RESET_THRESHOLD * recent_peak:
                self.alpha *= self._RESET_FACTOR  # OFI fast-reset

        self.prev_bid, self.prev_ask = bid, ask
        self.prev_bv, self.prev_av = bv, av
        return self.alpha

    def to_dict(self) -> Dict[str, Any]:
        return {"ema": self.ema, "prev_bid": self.prev_bid, "prev_ask": self.prev_ask,
                "prev_bv": self.prev_bv, "prev_av": self.prev_av, "alpha": self.alpha,
                # MODIFIED [1]: persist fast-reset window across ticks.
                "peak_buf": list(self.peak_buf)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OFIState":
        obj = cls(d.get("ema", 0.25))
        obj.prev_bid = d.get("prev_bid"); obj.prev_ask = d.get("prev_ask")
        obj.prev_bv = float(d.get("prev_bv", 0.0)); obj.prev_av = float(d.get("prev_av", 0.0))
        obj.alpha = float(d.get("alpha", 0.0))
        # MODIFIED [1]: restore rolling peak buffer (default empty for backwards compat).
        obj.peak_buf = [float(x) for x in d.get("peak_buf", [])]
        return obj


# ---------------------------------------------------------------------------
# Helper pricing (invariati vs S20)
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    if x >= 0:
        e = math.exp(-x); return 1.0 / (1.0 + e)
    e = math.exp(x); return e / (1.0 + e)


def _clamp(x: float, cap: float) -> float:
    return max(-cap, min(cap, float(x)))


def compute_microprice(bid: float, ask: float, bv: float, av: float) -> float:
    bv = max(float(bv), 1.0); av = max(float(av), 1.0)
    return (float(bid) * av + float(ask) * bv) / (av + bv)


def compute_queue_imbalance_depth(buys, sells, levels: int = 3) -> float:
    bs = sorted(buys.items(), key=lambda kv: -kv[0])[:levels]
    asks = sorted(sells.items(), key=lambda kv: kv[0])[:levels]
    bv = sum(max(1, int(v)) for _, v in bs)
    av = sum(max(1, int(-v)) for _, v in asks)
    if bv + av <= 0:
        return 0.0
    return (bv - av) / (bv + av)


def smooth_stress(z: float, inv_frac_abs: float, params: Dict[str, Any]) -> float:
    s_z = _sigmoid(float(params["stress_z_k"]) * (z - float(params["stress_z_center"])))
    s_q = _sigmoid(float(params["stress_inv_k"]) * (inv_frac_abs - float(params["stress_inv_center"])))
    return 1.0 - (1.0 - s_z) * (1.0 - s_q)


def inventory_price_skew(inventory: int, params: Dict[str, Any]) -> float:
    if not bool(params.get("asym_skew_v2", True)):
        return 0.0
    k = float(params.get("inv_skew_price", 0.0))
    cap = float(params.get("inv_shift_cap", 5.0))

    # STRATEGY25: keep the S23 backbone but make the price skew less reactive
    # near flat inventory, and only mildly steeper once inventory becomes large.
    if not bool(params.get("skew_piecewise", False)):
        return _clamp(-k * float(inventory), cap)

    inv_lim = max(1, int(params.get("inv_limit", 1)))
    inv_f = float(inventory)
    inv_abs = abs(inv_f)
    r = inv_abs / inv_lim

    dband = float(params.get("skew_deadband_frac", 0.10))
    kick = float(params.get("skew_kickin_frac", 0.60))
    mult = float(params.get("skew_kickin_mult", 0.75))

    if inv_f == 0.0 or r <= dband:
        return 0.0

    sign = 1.0 if inv_f > 0.0 else -1.0
    excess_abs = inv_abs - dband * inv_lim

    if r <= kick:
        skew = -sign * k * excess_abs
    else:
        extra_frac = max(0.0, r - kick)
        skew = -sign * k * excess_abs * (1.0 + mult * extra_frac)

    return _clamp(skew, cap)

def inventory_shift(inventory: int, z: float, params: Dict[str, Any],
                    side: Optional[str] = None, time_left: float = 1.0) -> float:
    q = float(inventory)
    if q == 0.0:
        return 0.0

    inv_lim = max(1, int(params["inv_limit"]))
    inv_frac = abs(q) / inv_lim
    use_v2 = bool(params.get("asym_skew_v2", True))

    base = 0.0 if use_v2 else float(params["inv_skew"]) * q

    # MODIFIED [3 — inventory danger zone]: above 70% of limit, scale
    # inv_extra up (more aggressive unloading) and inv_hard_frac down (act sooner).
    # Also scale with time pressure (fuller aggressiveness as session runs out).
    INV_DANGER = 0.70
    inv_extra_val = float(params["inv_extra"])
    hard_frac = float(params["inv_hard_frac"])
    if inv_frac >= INV_DANGER:
        danger_scale = min(1.0, (inv_frac - INV_DANGER) / (1.0 - INV_DANGER))
        inv_extra_dynamic = inv_extra_val * (1.0 + 2.0 * danger_scale)
        inv_hard_frac_dynamic = max(hard_frac * 0.5, hard_frac * (1.0 - 0.3 * danger_scale))
        # time pressure: time_left in [0,1]; ramp up as time_left->0
        tl = max(0.0, min(1.0, float(time_left)))
        time_pressure = 1.0 + max(0.0, 1.0 - tl) * 1.5
        inv_extra_dynamic *= time_pressure
    else:
        inv_extra_dynamic = inv_extra_val
        inv_hard_frac_dynamic = hard_frac

    extra = 0.0
    if inv_frac > inv_hard_frac_dynamic:
        extra = inv_extra_dynamic * (inv_frac - inv_hard_frac_dynamic) * q

    s = smooth_stress(z, inv_frac, params)
    raw = -(base + s * extra)
    raw = _clamp(raw, float(params["inv_shift_cap"]))

    if bool(params.get("asym_skew", True)) and side is not None:
        if q > 0 and side == "bid":
            return 0.0
        if q < 0 and side == "ask":
            return 0.0

    return raw


def inventory_close_side_bonus(inventory: int, params: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """MODIFIED [3 — close-side bonus]: extra tightening (in price units) applied
    to the side that REDUCES inventory when in the danger zone.
    Returns (bonus, side) where side in {"ask","bid",None}.
    - If long (q>0): close side is ask (sell); return positive bonus for ask half-spread tightening.
    - If short (q<0): close side is bid (buy); return positive bonus for bid half-spread tightening.
    """
    q = float(inventory)
    if q == 0.0:
        return 0.0, None
    inv_lim = max(1, int(params["inv_limit"]))
    inv_frac = abs(q) / inv_lim
    INV_DANGER = 0.70
    if inv_frac < INV_DANGER:
        return 0.0, None
    danger_scale = min(1.0, (inv_frac - INV_DANGER) / (1.0 - INV_DANGER))
    bonus = danger_scale * float(params["min_half"])
    close_side = "ask" if q > 0 else "bid"
    return bonus, close_side


def reservation_core(mid: float, micro_dev: float, imbalance: float, mkt_half: float,
                     params: Dict[str, Any], regime: str) -> float:
    cap = float(params["signal_cap"])
    if regime == "high":
        cap *= 1.10
    elif regime == "low":
        cap *= 0.90

    # MODIFIED [1 — signal_cap dynamic]: do not let directional signal push the
    # reservation more than half the current spread away from mid (mkt_half is
    # exactly half the spread). Keeps quotes from wandering past the opposite best.
    signal_cap_dynamic = min(cap, 0.5 * 2.0 * max(1.0, float(mkt_half)))
    # Note: 2.0*mkt_half == current_spread; 0.5*spread == mkt_half. Kept explicit
    # for symmetry with the spec formula "0.5 * current_spread".
    cap = signal_cap_dynamic

    micro_norm = float(micro_dev) / max(1.0, mkt_half)
    micro_c = _clamp(float(params["w_micro"]) * micro_norm, cap)
    imb_c = _clamp(float(params["w_imb"]) * float(imbalance), cap)
    return float(mid) + micro_c + imb_c


def quote_half_spreads(market_half: float, regime: str, inventory: int, params: Dict[str, Any],
                       min_half_override: Optional[float] = None) -> Tuple[float, float]:
    # MODIFIED [2 — dynamic min_half]: callers may pass min_half_override computed
    # from rolling spread median * (1 + 0.3 * vol_ratio). When None, behavior
    # matches the original (params["min_half"] is used).
    base_min_half = float(params["min_half"]) if min_half_override is None else float(min_half_override)

    base = max(base_min_half, market_half - float(params["touch_offset"]))
    if regime == "high":
        base += float(params["hi_widen"])
    elif regime == "low":
        base = max(base_min_half, base - float(params["lo_tighten"]))

    inv_lim = max(1, int(params["inv_limit"]))
    inv_frac = float(inventory) / inv_lim
    bid_h = base
    ask_h = base
    widen = float(params["inv_widen"])
    use_v2 = bool(params.get("asym_skew_v2", True))

    if use_v2:
        if inv_frac > 0:
            bid_h += widen * inv_frac
        elif inv_frac < 0:
            ask_h += widen * (-inv_frac)
    else:
        if inv_frac > 0:
            bid_h += widen * inv_frac
        elif inv_frac < 0:
            ask_h += widen * (-inv_frac)

    return bid_h, ask_h


def quote_size_for_side(side: str, inventory: int, regime: str, params: Dict[str, Any]) -> int:
    base = max(1, int(params["size"]))
    mult = float(params["hi_size_mult"]) if regime == "high" else float(params["lo_size_mult"])
    base = max(1, int(round(base * mult)))

    inv_lim = max(1, int(params["inv_limit"]))
    soft = int(params.get("inv_soft", inv_lim))
    q = int(inventory)

    if side == "bid":
        if q >= soft:
            return 0
        if q >= 0:
            frac = min(1.0, q / inv_lim)
            if frac >= 0.9:
                return 0
            return max(0, int(round(base * (1.0 - frac))))
        return max(0, min(base, inv_lim - q))

    if side == "ask":
        if q <= -soft:
            return 0
        if q <= 0:
            frac = min(1.0, -q / inv_lim)
            if frac >= 0.9:
                return 0
            return max(0, int(round(base * (1.0 - frac))))
        return max(0, min(base, inv_lim + q))

    return 0


def apply_best_edge(bid_q: int, ask_q: int, best_bid: int, best_ask: int,
                    mkt_spread: int, params: Dict[str, Any],
                    spread_regime: str = "legacy",
                    imbalance: Optional[float] = None,
                    ofi_alpha: Optional[float] = None) -> Tuple[int, int, Optional[str], Optional[str]]:
    # STRATEGY25:
    # - retain S23 gating by rolling spread regime
    # - import S24 flow-aware best-edge logic
    if spread_regime == "legacy":
        threshold = int(params.get("wide_spread_tick", 999_999))
        if mkt_spread < threshold:
            return bid_q, ask_q, None, None
    else:
        if spread_regime != "wide":
            return bid_q, ask_q, None, None

    offset = int(params.get("best_edge_offset", 0))
    target_bid = best_bid - offset
    target_ask = best_ask + offset

    if not bool(params.get("best_edge_flow_aware", False)):
        new_bid = max(bid_q, target_bid)
        new_ask = min(ask_q, target_ask)
        return new_bid, new_ask, None, None

    w_imb = float(params.get("w_imb_flow", 0.55))
    w_ofi = float(params.get("w_ofi_flow", 0.45))
    gate = float(params.get("flow_gate", 0.10))
    caut_off = int(params.get("cautious_offset", 1))
    neutral = str(params.get("best_edge_neutral_mode", "cautious"))

    imb = float(imbalance) if imbalance is not None else 0.0
    alpha = float(ofi_alpha) if ofi_alpha is not None else 0.0
    flow_score = w_imb * imb + w_ofi * alpha

    fav_bid = flow_score > gate
    fav_ask = flow_score < -gate
    is_neutral = (not fav_bid) and (not fav_ask)

    if fav_bid or (is_neutral and neutral == "join"):
        bid_target = target_bid
        bid_mode = "join"
    else:
        bid_target = target_bid - caut_off
        bid_mode = "cautious"

    if fav_ask or (is_neutral and neutral == "join"):
        ask_target = target_ask
        ask_mode = "join"
    else:
        ask_target = target_ask + caut_off
        ask_mode = "cautious"

    new_bid = max(bid_q, bid_target)
    new_ask = min(ask_q, ask_target)
    return new_bid, new_ask, bid_mode, ask_mode


# MODIFIED [2 — rolling spread regime]: helper used from Trader.run().
# No new BASE_PARAMS keys; constants hardcoded below.
_SPREAD_BUF_MAX: int = 200
_SPREAD_REGIME_WARMUP: int = 20  # below this, behave as "normal"


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    idx = p * (len(s) - 1)
    lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def classify_spread_regime(spread_buf: Sequence[float], current_spread: float) -> Tuple[str, float, float, float]:
    """Classify the current spread vs rolling distribution.
    Returns (regime, p25, p50, p75). Regime in {"wide","tight","normal"}.
    If the buffer is shorter than warmup, regime is "normal"."""
    if len(spread_buf) < _SPREAD_REGIME_WARMUP:
        med = current_spread if not spread_buf else _percentile(spread_buf, 0.5)
        return "normal", current_spread, med, current_spread
    p25 = _percentile(spread_buf, 0.25)
    p50 = _percentile(spread_buf, 0.50)
    p75 = _percentile(spread_buf, 0.75)
    if current_spread > p75:
        regime = "wide"
    elif current_spread < p25:
        regime = "tight"
    else:
        regime = "normal"
    return regime, p25, p50, p75


def sanity_clamp_quotes(bid_quote: int, ask_quote: int, best_bid: int, best_ask: int) -> Tuple[int, int, bool]:
    would_cross = (bid_quote >= best_ask) or (ask_quote <= best_bid) or (ask_quote <= bid_quote)
    bq = min(int(bid_quote), int(best_ask) - 1)
    aq = max(int(ask_quote), int(best_bid) + 1)
    if aq <= bq:
        aq = bq + 1
    return bq, aq, would_cross


def pepper_ofi_adjustment(ofi_alpha: float, mkt_half: float, params: Dict[str, Any]) -> float:
    raw = float(params["w_ofi"]) * float(ofi_alpha) * max(1.0, float(mkt_half))
    return _clamp(raw, float(params["ofi_cap"]))


def apply_pepper_signal_policy_v2(
    core: float,
    ofi_alpha: float,
    bid_half: float,
    ask_half: float,
    bid_size: int,
    ask_size: int,
    params: Dict[str, Any],
) -> Tuple[float, float, float, int, int]:
    gate = float(params["pepper_alpha_gate"])
    one_side = float(params["pepper_one_sided_gate"])
    jt = float(params["pepper_join_tighten"])
    lw = float(params["pepper_lean_widen"])
    min_half = float(params["min_half"])

    if ofi_alpha >= gate:
        ask_half = max(min_half, ask_half - jt)
        bid_half = bid_half + lw
        bid_size = max(0, bid_size // 2)
    elif ofi_alpha <= -gate:
        bid_half = max(min_half, bid_half - jt)
        ask_half = ask_half + lw
        ask_size = max(0, ask_size // 2)

    if ofi_alpha >= one_side:
        bid_size = 0
    elif ofi_alpha <= -one_side:
        ask_size = 0

    return core, bid_half, ask_half, bid_size, ask_size


def pepper_inventory_impulse_orders(
    product: str, inv: int, ofi_alpha: float,
    best_bid: int, best_ask: int, bv: float, av: float,
    params: Dict[str, Any],
):
    out = []
    if Order is None:
        return out, inv

    gate = float(params["pepper_impulse_gate"])
    qty0 = int(params["pepper_impulse_qty"])

    if inv < 0 and ofi_alpha >= gate:
        qty = min(qty0, -inv, int(av))
        if qty > 0:
            out.append(Order(product, int(best_ask), int(qty)))
            inv += qty
    elif inv > 0 and ofi_alpha <= -gate:
        qty = min(qty0, inv, int(bv))
        if qty > 0:
            out.append(Order(product, int(best_bid), -int(qty)))
            inv -= qty

    return out, inv


try:
    from datamodel import Order, TradingState  # type: ignore
except Exception:
    Order = None  # type: ignore
    TradingState = None  # type: ignore


# ===========================================================================
# (A) DIAGNOSTICS MODULE
#
# L'intero modulo e' OPT-IN: Trader usa self._diag.enabled come guard veloce;
# quando disabilitato i path non toccano strutture extra. Nessun nuovo parametro
# di strategia: la diagnostica vive in un wrapper esterno al Trader.
#
# Tag supportati per gli ordini:
#   - "base_make"           quota passiva non toccata da best-edge
#   - "best_edge"           quota passiva ancorata al best (spread wide)
#   - "take_mode"           fill aggressivo da logica take (edge_buy/edge_sell)
#   - "pepper_impulse"      ordine di riduzione inventario contro OFI impulso
#   - "eos_liquidation"     ordine di fine sessione (time_left < liq_time_frac)
#   - "unknown_flatten"     flatten passivo su prodotti non-noti con book mono-sided
#   - "sanity_cross_skip"   make soppressa da sanity_clamp con would_cross=True
#
# Ogni TickSnapshot registra i counterfactual necessari per l'attribuzione:
#   - raw_bid / raw_ask  prima di best-edge
#   - post_bid / post_ask dopo best-edge
#   - used_best_edge bool
#   - skew_shift (inventory_price_skew applicato al core)
#   - would_cross bool (dal sanity_clamp finale)
#   - ofi_alpha, regime, z, mkt_spread per stratificare
#
# Queste info sono sufficienti per la (1) attribution senza rigirare il
# backtest: basta matchare fill->order usando (ts, product, price, qty, tag).
# ===========================================================================

@dataclass
class TaggedOrder:
    ts: int
    tick: int
    product: str
    price: int
    qty: int          # qty firmata: + = buy, - = sell
    tag: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts, "tick": self.tick, "product": self.product,
            "price": self.price, "qty": self.qty, "tag": self.tag,
            "extra": dict(self.extra),
        }


@dataclass
class TickSnapshot:
    ts: int
    tick: int
    product: str
    mid: float
    best_bid: int
    best_ask: int
    mkt_spread: int
    regime: str
    z: float
    ofi_alpha: float
    inv_before: int
    # quote geometry counterfactual
    raw_bid: Optional[int] = None
    raw_ask: Optional[int] = None
    post_bid: Optional[int] = None   # dopo best-edge
    post_ask: Optional[int] = None
    final_bid: Optional[int] = None  # dopo sanity clamp
    final_ask: Optional[int] = None
    used_best_edge: bool = False
    would_cross: bool = False
    skew_shift: float = 0.0          # inventory_price_skew applicato al core
    pepper_ofi_adj: float = 0.0
    pepper_policy_triggered: bool = False
    time_left: float = 1.0
    liq_active: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}


@dataclass
class Diagnostics:
    """Container opzionale per instrumentare il Trader.

    Usage:
        diag = Diagnostics(enabled=True)
        Trader.diagnostics = diag       # class-level hook (o istanza)
        # ... run backtest ...
        diag.dump("diag_run1.json")

    Nessun impatto sul match engine: Diagnostics riceve una copia dei dati,
    gli ordini restituiti dal Trader sono gli stessi di S20.
    """
    enabled: bool = True
    order_log: List[TaggedOrder] = field(default_factory=list)
    ticks: List[TickSnapshot] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.order_log.clear()
        self.ticks.clear()
        self.counters.clear()

    def bump(self, key: str, n: int = 1) -> None:
        if not self.enabled:
            return
        self.counters[key] = self.counters.get(key, 0) + n

    def log_order(self, ts: int, tick: int, product: str, price: int, qty: int,
                  tag: str, **extra: Any) -> None:
        if not self.enabled:
            return
        self.order_log.append(TaggedOrder(ts, tick, product, int(price), int(qty), tag,
                                          dict(extra)))
        self.bump(f"orders:{tag}:{product}")

    def log_tick(self, snap: TickSnapshot) -> None:
        if not self.enabled:
            return
        self.ticks.append(snap)

    def dump(self, path: str) -> None:
        payload = {
            "counters": dict(self.counters),
            "orders": [o.as_dict() for o in self.order_log],
            "ticks": [t.as_dict() for t in self.ticks],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "Diagnostics":
        with open(path, "r") as fh:
            payload = json.load(fh)
        obj = cls(enabled=False)
        obj.counters = dict(payload.get("counters", {}))
        for o in payload.get("orders", []):
            obj.order_log.append(TaggedOrder(
                ts=int(o["ts"]), tick=int(o["tick"]), product=o["product"],
                price=int(o["price"]), qty=int(o["qty"]), tag=o["tag"],
                extra=dict(o.get("extra", {})),
            ))
        for t in payload.get("ticks", []):
            obj.ticks.append(TickSnapshot(**t))
        return obj


# ---------------------------------------------------------------------------
# Symbol state / (de)serialize (invariati vs S20)
# ---------------------------------------------------------------------------
@dataclass
class SymbolState:
    sv: Dict[str, Any] = field(default_factory=dict)
    ofi: Dict[str, Any] = field(default_factory=dict)
    last_mid: Optional[float] = None
    tick: int = 0
    session_max_ts: int = 0
    # MODIFIED [2]: rolling spread distribution for regime classification.
    spread_buf: List[float] = field(default_factory=list)
    # MODIFIED [1/2]: baseline volatility reference (ewa_abs_r at warmup).
    sigma_ref: Optional[float] = None


def _serialize(states: Dict[str, SymbolState]) -> str:
    return json.dumps({
        k: {"sv": v.sv, "ofi": v.ofi, "last_mid": v.last_mid,
            "tick": v.tick, "session_max_ts": v.session_max_ts,
            # MODIFIED [2]: persist rolling spread buffer and sigma_ref.
            "spread_buf": list(v.spread_buf), "sigma_ref": v.sigma_ref}
        for k, v in states.items()
    })


def _deserialize(blob: str) -> Dict[str, SymbolState]:
    if not blob:
        return {}
    try:
        raw = json.loads(blob)
    except Exception:
        return {}
    out: Dict[str, SymbolState] = {}
    for k, v in raw.items():
        out[k] = SymbolState(
            sv=v.get("sv", {}), ofi=v.get("ofi", {}),
            last_mid=v.get("last_mid"),
            tick=int(v.get("tick", 0)),
            session_max_ts=int(v.get("session_max_ts", 0)),
            # MODIFIED [2]: restore rolling spread buffer (default empty) and sigma_ref.
            spread_buf=[float(x) for x in v.get("spread_buf", [])],
            sigma_ref=v.get("sigma_ref", None),
        )
    return out


# ---------------------------------------------------------------------------
# TRADER (logica S20 preservata; diff = solo hook di diagnostica + overrides)
# ---------------------------------------------------------------------------
class Trader:
    # Hook class-level per il backtester. Se None -> nessun overhead.
    diagnostics: Optional[Diagnostics] = None
    # Hook class-level per sensitivity: overrides globali {product: {key: value}}
    param_overrides_runtime: Optional[Dict[str, Dict[str, Any]]] = None

    def __init__(self) -> None:
        self._states: Dict[str, SymbolState] = {}
        self._svs: Dict[str, HullWhiteSV] = {}
        self._ofis: Dict[str, OFIState] = {}
        # instance-level fallback: se assegnato, ha la precedenza sulla class-level
        self._diag_local: Optional[Diagnostics] = None
        self._overrides_local: Optional[Dict[str, Dict[str, Any]]] = None

    # --- Diagnostics plumbing ------------------------------------------------
    @property
    def _diag(self) -> Optional[Diagnostics]:
        return self._diag_local if self._diag_local is not None else Trader.diagnostics

    def attach_diagnostics(self, diag: Diagnostics) -> None:
        self._diag_local = diag

    def attach_overrides(self, overrides: Dict[str, Dict[str, Any]]) -> None:
        self._overrides_local = overrides

    def _params(self, product: str) -> Dict[str, Any]:
        runtime_overrides = self._overrides_local if self._overrides_local is not None \
            else Trader.param_overrides_runtime
        ov = None
        if runtime_overrides:
            ov = runtime_overrides.get(product)
        return params_for(product, overrides=ov)

    def _emit(self, orders: List[Any], product: str, price: int, qty: int,
              tag: str, ts: int, tick: int, **extra: Any) -> None:
        """Crea l'Order e lo appende a `orders`; in parallelo logga il tag.
        Qty firmata: + = buy, - = sell. Questo e' lo stesso formato di S20.
        """
        if Order is None:
            return
        orders.append(Order(product, int(price), int(qty)))
        diag = self._diag
        if diag is not None and diag.enabled:
            diag.log_order(ts, tick, product, int(price), int(qty), tag, **extra)

    # --- State bookkeeping ---------------------------------------------------
    def _ensure(self, product: str) -> Tuple[SymbolState, HullWhiteSV, OFIState]:
        st = self._states.get(product)
        if st is None:
            st = SymbolState(); self._states[product] = st

        sv = self._svs.get(product)
        if sv is None:
            p = self._params(product)
            if st.sv:
                try:
                    sv = HullWhiteSV.from_dict(st.sv)
                except Exception:
                    sv = HullWhiteSV(
                        p["hw_kappa"], p["hw_q"], p["hw_R"],
                        p["hw_z_hi"], p["hw_z_lo"], int(p["hw_warmup"]),
                        p.get("hw_tc_kappa", 0.18), p.get("hw_tc_theta", 1.0),
                        p.get("hw_tc_jump_alpha", 0.18), p.get("hw_tc_jump_gate", 2.2),
                        p.get("hw_tc_jump_cap", 4.0), p.get("hw_tc_q_mult", 1.25),
                        p.get("hw_tc_R_div", 0.70), p.get("hw_tc_x_jump", 0.22),
                        p.get("hw_tc_tau_min", 0.75), p.get("hw_tc_tau_max", 3.50),
                        p.get("hw_absret_alpha", 0.08)
                    )
            else:
                sv = HullWhiteSV(p["hw_kappa"], p["hw_q"], p["hw_R"],
                                 p["hw_z_hi"], p["hw_z_lo"], int(p["hw_warmup"]))
            self._svs[product] = sv

        ofi = self._ofis.get(product)
        if ofi is None:
            p = self._params(product)
            if st.ofi:
                try:
                    ofi = OFIState.from_dict(st.ofi)
                except Exception:
                    ofi = OFIState(float(p.get("ofi_ema", 0.25)))
            else:
                ofi = OFIState(float(p.get("ofi_ema", 0.25)))
            self._ofis[product] = ofi

        return st, sv, ofi

    def _save(self) -> str:
        for sym, st in self._states.items():
            if sym in self._svs:
                st.sv = self._svs[sym].to_dict()
            if sym in self._ofis:
                st.ofi = self._ofis[sym].to_dict()
        return _serialize(self._states)

    def _estimate_time_left(self, st: SymbolState, ts: int, params: Dict[str, Any]) -> float:
        if ts > st.session_max_ts:
            st.session_max_ts = ts
        session_len = max(int(params.get("session_len_default", 200_000)),
                          st.session_max_ts)
        return 1.0 - (ts / max(1, session_len))

    # --- Main loop -----------------------------------------------------------
    def run(self, state: "TradingState"):
        self._states = _deserialize(getattr(state, "traderData", "") or "")
        self._svs = {}
        self._ofis = {}

        result: Dict[str, List[Any]] = {}
        order_depths = getattr(state, "order_depths", {}) or {}
        positions = getattr(state, "position", {}) or {}
        ts = int(getattr(state, "timestamp", 0))
        diag = self._diag

        for product, depth in order_depths.items():
            params = self._params(product)
            is_known = product in PARAM_OVERRIDES
            orders: List[Any] = []
            buys = getattr(depth, "buy_orders", {}) or {}
            sells = getattr(depth, "sell_orders", {}) or {}

            if not buys or not sells:
                inv = int(positions.get(product, 0))
                tick_id = self._states.get(product, SymbolState()).tick
                if not is_known and Order is not None and inv != 0:
                    if inv > 0 and buys:
                        bb = max(buys.keys())
                        self._emit(orders, product, int(bb), -1,
                                   tag="unknown_flatten", ts=ts, tick=tick_id)
                    elif inv < 0 and sells:
                        ba = min(sells.keys())
                        self._emit(orders, product, int(ba), 1,
                                   tag="unknown_flatten", ts=ts, tick=tick_id)
                result[product] = orders
                continue

            best_bid = max(buys.keys())
            best_ask = min(sells.keys())
            bv = float(buys[best_bid])
            av = float(-sells[best_ask])
            mkt_spread = int(best_ask - best_bid)

            mid = 0.5 * (best_bid + best_ask)
            microp = compute_microprice(best_bid, best_ask, bv, av)
            micro_dev = microp - mid
            imbalance = compute_queue_imbalance_depth(buys, sells, levels=3)

            st, sv, ofi = self._ensure(product)
            # MODIFIED [4/1]: run sv.update FIRST so that its fresh ewa_abs_r can
            # feed the vol_ratio passed into ofi.update. Ordering swap does not
            # change the UKF computation itself (it reads only `mid`).
            _, z, regime = sv.update(mid)

            # MODIFIED [1 — vol_ratio]: short-horizon realized-vol ratio vs baseline.
            # Baseline sigma_ref seeded from ewa_abs_r once the UKF exits warmup;
            # until then vol_ratio == 1.0 (identical to S20 behavior).
            if st.sigma_ref is None and sv.n >= int(params.get("hw_warmup", 30)):
                st.sigma_ref = max(0.25, float(sv.ewa_abs_r))
            if st.sigma_ref is not None and st.sigma_ref > 1e-9:
                vol_ratio = min(3.0, float(sv.ewa_abs_r) / float(st.sigma_ref))
            else:
                vol_ratio = 1.0

            # MODIFIED [1]: pass vol_ratio to OFI; default kwarg keeps OFIState
            # call-compatible with any other caller.
            ofi_alpha = ofi.update(best_bid, bv, best_ask, av, vol_ratio=vol_ratio)

            # MODIFIED [2 — rolling spread regime]: update buffer BEFORE classifying.
            st.spread_buf.append(float(mkt_spread))
            if len(st.spread_buf) > _SPREAD_BUF_MAX:
                st.spread_buf.pop(0)
            spread_regime, _sp25, spread_p50, _sp75 = classify_spread_regime(
                st.spread_buf, float(mkt_spread))

            st.last_mid = mid
            st.tick += 1
            tick_id = st.tick

            inv = int(positions.get(product, 0))
            inv_before = inv
            mkt_half = 0.5 * (best_ask - best_bid)

            # MODIFIED [3]: compute time_left earlier so inventory_shift can see it.
            time_left = self._estimate_time_left(st, ts, params)

            core_pre = reservation_core(mid, micro_dev, imbalance, mkt_half, params, regime)
            skew_shift = inventory_price_skew(inv, params)
            core = core_pre + skew_shift

            pepper_ofi_adj = 0.0
            if product == "INTARIAN_PEPPER_ROOT":
                pepper_ofi_adj = pepper_ofi_adjustment(ofi_alpha, mkt_half, params)
                core += pepper_ofi_adj

            # MODIFIED [3]: time-pressure enters inventory_shift via time_left kwarg.
            bid_center = core + inventory_shift(inv, z, params, side="bid", time_left=time_left)
            ask_center = core + inventory_shift(inv, z, params, side="ask", time_left=time_left)

            # MODIFIED [2 — dynamic min_half]: scale floor with rolling median spread
            # and short-horizon vol. Clamp to [base, 2.5*base].
            min_half_base = float(params["min_half"])
            spread_ref = max(1.0, 2.0 * min_half_base)  # calibrated baseline spread
            vol_boost = 1.0 + 0.3 * max(0.0, vol_ratio - 1.0)
            min_half_dynamic = min_half_base * (max(1.0, spread_p50) / spread_ref) * vol_boost
            min_half_dynamic = max(min_half_base, min(min_half_base * 2.5, min_half_dynamic))

            bid_half, ask_half = quote_half_spreads(
                mkt_half, regime, inv, params, min_half_override=min_half_dynamic)

            # MODIFIED [3 — close-side bonus]: extra tightening on the side that
            # reduces inventory, only active in the danger zone (>=70% of limit).
            cs_bonus, close_side = inventory_close_side_bonus(inv, params)
            if cs_bonus > 0.0 and close_side == "ask":
                ask_half = max(min_half_dynamic, ask_half - cs_bonus)
            elif cs_bonus > 0.0 and close_side == "bid":
                bid_half = max(min_half_dynamic, bid_half - cs_bonus)

            # MODIFIED [2]: in "tight" spread regime, step back by best_edge_offset
            # on both sides (never narrower than the dynamic min_half floor).
            if spread_regime == "tight":
                be_off = float(params.get("best_edge_offset", 0))
                if be_off > 0:
                    bid_half = max(min_half_dynamic, bid_half + be_off)
                    ask_half = max(min_half_dynamic, ask_half + be_off)

            # --- (4) Liquidazione di fine sessione ---
            # time_left was already computed above (moved earlier for MOD 3).
            liq_active = False
            if Order is not None and time_left < float(params["liq_time_frac"]) \
                    and abs(inv) >= int(params["liq_min_inv"]):
                liq_active = True
                chunk = int(params["liq_chunk"])
                if inv > 0:
                    qty = min(chunk, inv, int(bv))
                    if qty > 0:
                        self._emit(orders, product, int(best_bid), -int(qty),
                                   tag="eos_liquidation", ts=ts, tick=tick_id,
                                   time_left=time_left)
                        inv -= qty
                elif inv < 0:
                    qty = min(chunk, -inv, int(av))
                    if qty > 0:
                        self._emit(orders, product, int(best_ask), int(qty),
                                   tag="eos_liquidation", ts=ts, tick=tick_id,
                                   time_left=time_left)
                        inv += qty

            # --- Pepper impulse ---
            if product == "INTARIAN_PEPPER_ROOT" and Order is not None:
                gate = float(params["pepper_impulse_gate"])
                qty0 = int(params["pepper_impulse_qty"])
                if inv < 0 and ofi_alpha >= gate:
                    qty = min(qty0, -inv, int(av))
                    if qty > 0:
                        self._emit(orders, product, int(best_ask), int(qty),
                                   tag="pepper_impulse", ts=ts, tick=tick_id,
                                   ofi_alpha=ofi_alpha)
                        inv += qty
                elif inv > 0 and ofi_alpha <= -gate:
                    qty = min(qty0, inv, int(bv))
                    if qty > 0:
                        self._emit(orders, product, int(best_bid), -int(qty),
                                   tag="pepper_impulse", ts=ts, tick=tick_id,
                                   ofi_alpha=ofi_alpha)
                        inv -= qty

            # --- TAKE ---
            if Order is not None and bool(params.get("take_enable", True)):
                inv_lim = int(params["inv_limit"])

                edge_buy = core - best_ask - float(params["cross_cost"]) - float(params["impact_coef"])
                if inv < inv_lim and edge_buy >= float(params["min_take_edge"]):
                    qty = min(int(params["size"]), int(av), inv_lim - inv)
                    if qty > 0:
                        self._emit(orders, product, int(best_ask), int(qty),
                                   tag="take_mode", ts=ts, tick=tick_id,
                                   edge=edge_buy, side="buy", core=core)
                        inv += qty
                else:
                    # counterfactual: take che NON e' scattato pur essendo abilitato
                    if diag is not None and diag.enabled and inv < inv_lim:
                        diag.bump(f"take_skip_buy:{product}")

                edge_sell = best_bid - core - float(params["cross_cost"]) - float(params["impact_coef"])
                if inv > -inv_lim and edge_sell >= float(params["min_take_edge"]):
                    qty = min(int(params["size"]), int(bv), inv + inv_lim)
                    if qty > 0:
                        self._emit(orders, product, int(best_bid), -int(qty),
                                   tag="take_mode", ts=ts, tick=tick_id,
                                   edge=edge_sell, side="sell", core=core)
                        inv -= qty
                else:
                    if diag is not None and diag.enabled and inv > -inv_lim:
                        diag.bump(f"take_skip_sell:{product}")

            # --- MAKE ---
            raw_bid = raw_ask = post_bid = post_ask = final_bid = final_ask = None
            used_best_edge = False
            would_cross = False
            pepper_policy_triggered = False

            if Order is not None:
                bid_size = quote_size_for_side("bid", inv, regime, params)
                ask_size = quote_size_for_side("ask", inv, regime, params)

                if product == "INTARIAN_PEPPER_ROOT":
                    bs_before, as_before = bid_size, ask_size
                    bh_before, ah_before = bid_half, ask_half
                    core, bid_half, ask_half, bid_size, ask_size = apply_pepper_signal_policy_v2(
                        core, ofi_alpha, bid_half, ask_half, bid_size, ask_size, params
                    )
                    pepper_policy_triggered = (
                        bs_before != bid_size or as_before != ask_size
                        or bh_before != bid_half or ah_before != ask_half
                    )
                    if pepper_policy_triggered and diag is not None:
                        diag.bump(f"pepper_policy:{product}")
                    # MODIFIED [3]: keep time_left in the pepper re-shift too.
                    bid_center = core + inventory_shift(inv, z, params, side="bid", time_left=time_left)
                    ask_center = core + inventory_shift(inv, z, params, side="ask", time_left=time_left)

                # MODIFIED [5 — signal sanity clamp]: compute the "desired" passive
                # quotes BEFORE clamping to the opposite best, so we can detect and
                # log the clamp event. The existing min/max floors already guarantee
                # the no-cross invariant; this block only adds observability and an
                # explicit defensive reassertion.
                desired_bid_float = bid_center - bid_half
                desired_ask_float = ask_center + ask_half
                clamp_bid_triggered = desired_bid_float >= float(best_ask)
                clamp_ask_triggered = desired_ask_float <= float(best_bid)

                raw_bid = int(math.floor(min(desired_bid_float, best_ask - 1.0)))
                raw_ask = int(math.ceil(max(desired_ask_float, best_bid + 1.0)))

                # MODIFIED [5]: defensive hard reassertion (no-op if floors above held).
                if raw_bid >= int(best_ask):
                    raw_bid = int(best_ask) - 1
                    clamp_bid_triggered = True
                if raw_ask <= int(best_bid):
                    raw_ask = int(best_bid) + 1
                    clamp_ask_triggered = True

                if (clamp_bid_triggered or clamp_ask_triggered) and diag is not None and diag.enabled:
                    diag.bump(f"signal_clamp:{product}")
                    diag.log_order(ts, tick_id, product, int(best_bid), 0,
                                   tag="signal_clamp_event",
                                   ofi_alpha=ofi_alpha, mkt_spread=mkt_spread,
                                   core=core, bid_side=bool(clamp_bid_triggered),
                                   ask_side=bool(clamp_ask_triggered))

                # MODIFIED [2 — best-edge gated by regime classifier]
                # STRATEGY25: lighter S24 asymmetry, layered on top of the S23
                # adaptive quoting stack and danger-zone controls.
                inv_exec_active = False
                inv_exec_closing_side = None
                if bool(params.get("inv_exec_asym", False)) and inv != 0:
                    inv_exec_active = True
                    inv_lim_f = max(1, int(params["inv_limit"]))
                    inv_frac_abs = abs(float(inv) / inv_lim_f)
                    hard_frac = float(params.get("inv_hard_frac", 0.65))

                    c_shrink = float(params.get("closing_half_shrink", 0.30))
                    c_min = float(params.get("closing_half_min_mult", 0.65))
                    c_boost = float(params.get("closing_size_boost", 0.30))
                    o_widen = float(params.get("opening_half_widen", 0.12))
                    o_cut = float(params.get("opening_size_cut", 0.20))
                    o_min = float(params.get("opening_size_min_mult", 0.50))

                    closing_half_mult = max(c_min, 1.0 - c_shrink * inv_frac_abs)
                    if inv_frac_abs > hard_frac:
                        opening_half_mult = 1.0
                    else:
                        opening_half_mult = 1.0 + o_widen * inv_frac_abs

                    closing_size_mult = 1.0 + c_boost * inv_frac_abs
                    opening_size_mult = max(o_min, 1.0 - o_cut * inv_frac_abs)

                    if inv > 0:
                        inv_exec_closing_side = "ask"
                        ask_half = ask_half * closing_half_mult
                        ask_size = min(int(params["inv_limit"]),
                                       max(0, int(round(ask_size * closing_size_mult))))
                        bid_half = bid_half * opening_half_mult
                        bid_size = max(0, int(round(bid_size * opening_size_mult)))
                    else:
                        inv_exec_closing_side = "bid"
                        bid_half = bid_half * closing_half_mult
                        bid_size = min(int(params["inv_limit"]),
                                       max(0, int(round(bid_size * closing_size_mult))))
                        ask_half = ask_half * opening_half_mult
                        ask_size = max(0, int(round(ask_size * opening_size_mult)))

                    desired_bid_float = bid_center - bid_half
                    desired_ask_float = ask_center + ask_half
                    clamp_bid_triggered = desired_bid_float >= float(best_ask)
                    clamp_ask_triggered = desired_ask_float <= float(best_bid)
                    raw_bid = int(math.floor(min(desired_bid_float, best_ask - 1.0)))
                    raw_ask = int(math.ceil(max(desired_ask_float, best_bid + 1.0)))
                    if raw_bid >= int(best_ask):
                        raw_bid = int(best_ask) - 1
                        clamp_bid_triggered = True
                    if raw_ask <= int(best_bid):
                        raw_ask = int(best_bid) + 1
                        clamp_ask_triggered = True

                post_bid, post_ask, bid_flow_mode, ask_flow_mode = apply_best_edge(
                                                    raw_bid, raw_ask,
                                                    int(best_bid), int(best_ask),
                                                    mkt_spread, params,
                                                    spread_regime=spread_regime,
                                                    imbalance=imbalance,
                                                    ofi_alpha=ofi_alpha)
                used_best_edge_bid = (post_bid != raw_bid)
                used_best_edge_ask = (post_ask != raw_ask)
                used_best_edge = used_best_edge_bid or used_best_edge_ask

                bid_q, ask_q, would_cross = sanity_clamp_quotes(post_bid, post_ask,
                                                                 int(best_bid), int(best_ask))
                final_bid, final_ask = bid_q, ask_q

                if would_cross and not bool(params.get("take_enable", True)):
                    if bid_size > 0 and bid_q == int(best_ask) - 1:
                        bid_size = 0
                        if diag is not None:
                            diag.bump(f"sanity_cross_skip:{product}")
                    if ask_size > 0 and ask_q == int(best_bid) + 1:
                        ask_size = 0
                        if diag is not None:
                            diag.bump(f"sanity_cross_skip:{product}")

                def _resolve_tag(side_name: str, used_be: bool,
                                 flow_mode: Optional[str]) -> str:
                    if inv_exec_active:
                        if inv_exec_closing_side == side_name:
                            return "inv_exec_aggressive"
                        return "inv_exec_patient"
                    if flow_mode == "join":
                        return "best_edge_flow_join"
                    if flow_mode == "cautious":
                        return "best_edge_flow_cautious"
                    if used_be:
                        return "best_edge"
                    return "base_make"

                if bid_size > 0:
                    tag = _resolve_tag("bid", used_best_edge_bid, bid_flow_mode)
                    self._emit(orders, product, int(bid_q), int(bid_size),
                               tag=tag, ts=ts, tick=tick_id,
                               raw_bid=raw_bid, post_bid=post_bid,
                               skew_shift=skew_shift, regime=regime,
                               mkt_spread=mkt_spread, ofi_alpha=ofi_alpha,
                               would_cross=would_cross,
                               pepper_policy=pepper_policy_triggered,
                               flow_mode=bid_flow_mode,
                               inv_exec_active=inv_exec_active,
                               closing_side=inv_exec_closing_side)
                if ask_size > 0:
                    tag = _resolve_tag("ask", used_best_edge_ask, ask_flow_mode)
                    self._emit(orders, product, int(ask_q), -int(ask_size),
                               tag=tag, ts=ts, tick=tick_id,
                               raw_ask=raw_ask, post_ask=post_ask,
                               skew_shift=skew_shift, regime=regime,
                               mkt_spread=mkt_spread, ofi_alpha=ofi_alpha,
                               would_cross=would_cross,
                               pepper_policy=pepper_policy_triggered,
                               flow_mode=ask_flow_mode,
                               inv_exec_active=inv_exec_active,
                               closing_side=inv_exec_closing_side)

            # --- tick snapshot (solo se diag attiva) ---
            if diag is not None and diag.enabled:
                diag.log_tick(TickSnapshot(
                    ts=ts, tick=tick_id, product=product,
                    mid=mid, best_bid=int(best_bid), best_ask=int(best_ask),
                    mkt_spread=mkt_spread, regime=regime, z=z,
                    ofi_alpha=ofi_alpha, inv_before=inv_before,
                    raw_bid=raw_bid, raw_ask=raw_ask,
                    post_bid=post_bid, post_ask=post_ask,
                    final_bid=final_bid, final_ask=final_ask,
                    used_best_edge=used_best_edge, would_cross=would_cross,
                    skew_shift=skew_shift, pepper_ofi_adj=pepper_ofi_adj,
                    pepper_policy_triggered=pepper_policy_triggered,
                    time_left=time_left, liq_active=liq_active,
                ))

            result[product] = orders

        trader_data = self._save()
        return result, 0, trader_data


# ===========================================================================
# (C) UTILITY OFFLINE PER I 4 WORKSTREAM
#
# Le funzioni seguenti operano su UNA LISTA DI FILL (dicts) prodotta da un
# backtester esterno. Ogni fill e' uno dei seguenti schemi (accettati entrambi):
#
#   schema "slim":
#     {"ts": int, "tick": int, "product": str, "price": int,
#      "qty": int,            # firmato: + buy, - sell
#      "tag": str,            # dal TaggedOrder.tag
#      "mid": float}          # mid a tempo del fill (per mark-to-mid)
#
#   schema "full":
#     come slim + fields extra (inventory, regime, ...): ignorati se presenti
#
# La metrica PnL usata di default e' "spread-to-mid" per fill:
#
#     pnl_fill = qty * (mid - price)     # positivo = fill sotto mid (buy)
#                                         # o sopra mid (sell)
#
# Per un accounting realized end-of-session il caller dovrebbe marcare
# l'inventario residuo al mid finale (o al liquidation price) e sommarlo
# al totale. `attribute_pnl_by_tag` espone gia' un hook per quello.
# ===========================================================================

Fill = Dict[str, Any]


def _fill_pnl_mid(f: Fill) -> float:
    qty = float(f["qty"])
    mid = float(f.get("mid", 0.0))
    price = float(f["price"])
    return qty * (mid - price)


def _fill_notional(f: Fill) -> float:
    return abs(float(f["qty"])) * float(f["price"])


# ---------- Workstream 1: attribution per tag -----------------------------
def attribute_pnl_by_tag(
    fills: Sequence[Fill],
    products: Optional[Sequence[str]] = None,
    residual_inv_mark: Optional[Dict[str, Tuple[int, float]]] = None,
    pnl_fn: Callable[[Fill], float] = _fill_pnl_mid,
) -> Dict[str, Any]:
    """WS1 - scompone PnL (mid-mark) per tag, per prodotto e totale.

    residual_inv_mark: {product: (inventory_residuo, mark_price)} -> aggiunge
    una riga pseudo-tag "residual_inv" al report.
    """
    products = list(products) if products else sorted({f["product"] for f in fills})
    rows: Dict[Tuple[str, str], Dict[str, float]] = {}

    for f in fills:
        key = (f["product"], f.get("tag", "unknown"))
        r = rows.setdefault(key, {"fills": 0, "qty_abs": 0, "notional": 0.0, "pnl": 0.0})
        r["fills"] += 1
        r["qty_abs"] += abs(int(f["qty"]))
        r["notional"] += _fill_notional(f)
        r["pnl"] += pnl_fn(f)

    if residual_inv_mark:
        # Conservative: residual inv PnL = - inv * mark (relative to 0 cost basis in terms of mark).
        # Il caller dovrebbe passare un mark gia' al netto del cost basis se vuole una firma pulita.
        for prod, (inv, mark) in residual_inv_mark.items():
            key = (prod, "residual_inv")
            r = rows.setdefault(key, {"fills": 0, "qty_abs": 0, "notional": 0.0, "pnl": 0.0})
            r["pnl"] += float(inv) * float(mark)

    # by tag globale + per prodotto
    by_tag: Dict[str, Dict[str, float]] = {}
    by_product: Dict[str, Dict[str, float]] = {p: {} for p in products}
    grand = {"fills": 0, "qty_abs": 0, "notional": 0.0, "pnl": 0.0}

    for (prod, tag), r in rows.items():
        g = by_tag.setdefault(tag, {"fills": 0, "qty_abs": 0, "notional": 0.0, "pnl": 0.0})
        for k, v in r.items():
            g[k] += v
        by_product.setdefault(prod, {})[tag] = dict(r)
        for k, v in r.items():
            grand[k] += v

    # enrichment: pnl per fill
    def _ppfx(r):
        r["pnl_per_fill"] = (r["pnl"] / r["fills"]) if r["fills"] else 0.0
        return r

    by_tag = {t: _ppfx(dict(r)) for t, r in by_tag.items()}
    for p in by_product:
        by_product[p] = {t: _ppfx(dict(r)) for t, r in by_product[p].items()}

    return {
        "by_tag": by_tag,
        "by_product": by_product,
        "total": _ppfx(dict(grand)),
    }


# ---------- Workstream 3: per-product stats -------------------------------
def _fills_to_cum_pnl(fills: Sequence[Fill],
                     pnl_fn: Callable[[Fill], float] = _fill_pnl_mid) -> List[Tuple[int, float]]:
    cum = 0.0
    out: List[Tuple[int, float]] = []
    for f in sorted(fills, key=lambda x: (int(x["ts"]), int(x.get("tick", 0)))):
        cum += pnl_fn(f)
        out.append((int(f["ts"]), cum))
    return out


def _max_drawdown(curve: Sequence[Tuple[int, float]]) -> Tuple[float, int]:
    peak = -float("inf"); dd = 0.0; dd_time = 0; last_peak_ts = 0
    for ts, v in curve:
        if v > peak:
            peak = v; last_peak_ts = ts
        drop = peak - v
        if drop > dd:
            dd = drop; dd_time = ts - last_peak_ts
    return dd, dd_time


def _moments(xs: Sequence[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0}
    n = len(xs)
    mean = sum(xs) / n
    med = statistics.median(xs)
    if n < 2:
        return {"mean": mean, "median": med, "std": 0.0, "skew": 0.0, "kurt": 0.0}
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return {"mean": mean, "median": med, "std": 0.0, "skew": 0.0, "kurt": 0.0}
    m3 = sum((x - mean) ** 3 for x in xs) / n
    m4 = sum((x - mean) ** 4 for x in xs) / n
    skew = m3 / (std ** 3)
    kurt = m4 / (std ** 4) - 3.0
    return {"mean": mean, "median": med, "std": std, "skew": skew, "kurt": kurt}


def per_product_stats(
    fills: Sequence[Fill],
    inv_limits: Optional[Dict[str, int]] = None,
    inv_softs: Optional[Dict[str, int]] = None,
    inventory_path: Optional[Dict[str, Sequence[Tuple[int, int]]]] = None,
    pnl_fn: Callable[[Fill], float] = _fill_pnl_mid,
) -> Dict[str, Dict[str, Any]]:
    """WS3 - statistiche per prodotto (distribuzione PnL, drawdown, inventory).

    inv_limits / inv_softs: opzionali, usate se inventory_path e' fornito
    per calcolare % tempo sopra le soglie.

    inventory_path: {product: [(ts, inv), ...]}
    """
    out: Dict[str, Dict[str, Any]] = {}
    by_product: Dict[str, List[Fill]] = {}
    for f in fills:
        by_product.setdefault(f["product"], []).append(f)

    for product, pfills in by_product.items():
        pnl_series = [pnl_fn(f) for f in pfills]
        moms = _moments(pnl_series)
        hit_rate = (sum(1 for x in pnl_series if x > 0) / len(pnl_series)) if pnl_series else 0.0
        gross_win = sum(x for x in pnl_series if x > 0)
        gross_loss = -sum(x for x in pnl_series if x < 0)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        curve = _fills_to_cum_pnl(pfills, pnl_fn)
        mdd, mdd_time = _max_drawdown(curve)

        tag_counts: Dict[str, int] = {}
        for f in pfills:
            tag_counts[f.get("tag", "unknown")] = tag_counts.get(f.get("tag", "unknown"), 0) + 1
        fill_rate_passive = (
            tag_counts.get("base_make", 0)
            + tag_counts.get("best_edge", 0)
            + tag_counts.get("best_edge_flow_join", 0)
            + tag_counts.get("best_edge_flow_cautious", 0)
            + tag_counts.get("inv_exec_aggressive", 0)
            + tag_counts.get("inv_exec_patient", 0)
        )
        take_rate = tag_counts.get("take_mode", 0)

        inv_stats: Dict[str, Any] = {}
        if inventory_path and product in inventory_path:
            path = inventory_path[product]
            if path:
                abs_max = max(abs(i) for _, i in path)
                inv_stats["max_abs_inventory"] = abs_max
                lim = (inv_limits or {}).get(product)
                soft = (inv_softs or {}).get(product)
                n = len(path)
                if soft is not None:
                    inv_stats["pct_time_abs_gt_soft"] = sum(1 for _, i in path if abs(i) > soft) / n
                if lim is not None:
                    inv_stats["pct_time_abs_gt_80pct_lim"] = \
                        sum(1 for _, i in path if abs(i) > 0.8 * lim) / n

        out[product] = {
            "n_fills": len(pfills),
            "cum_pnl": curve[-1][1] if curve else 0.0,
            "moments": moms,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "max_drawdown": mdd,
            "time_in_drawdown_ts": mdd_time,
            "tag_counts": tag_counts,
            "passive_fills": fill_rate_passive,
            "take_fills": take_rate,
            "inventory": inv_stats,
            "cum_pnl_series": curve,
            "trade_pnl_series": pnl_series,
        }

    return out


def compare_products(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """WS3 - side-by-side + correlazione PnL intraday fra due prodotti."""
    products = list(stats.keys())
    side_by_side: Dict[str, Dict[str, Any]] = {}
    for p in products:
        s = stats[p]
        side_by_side[p] = {
            "n_fills": s["n_fills"],
            "cum_pnl": s["cum_pnl"],
            "mean_trade": s["moments"]["mean"],
            "std_trade": s["moments"]["std"],
            "skew_trade": s["moments"]["skew"],
            "kurt_trade": s["moments"]["kurt"],
            "hit_rate": s["hit_rate"],
            "profit_factor": s["profit_factor"],
            "max_drawdown": s["max_drawdown"],
            "passive_fills": s["passive_fills"],
            "take_fills": s["take_fills"],
        }

    corr = None
    if len(products) == 2:
        a, b = products
        sa, sb = stats[a], stats[b]
        if sa["cum_pnl_series"] and sb["cum_pnl_series"]:
            # campiona entrambe le curve su una griglia comune di ts
            ts_set = sorted({t for t, _ in sa["cum_pnl_series"]} &
                            {t for t, _ in sb["cum_pnl_series"]})
            if len(ts_set) >= 3:
                map_a = dict(sa["cum_pnl_series"])
                map_b = dict(sb["cum_pnl_series"])
                va = [map_a[t] for t in ts_set]
                vb = [map_b[t] for t in ts_set]
                # correlazione sui DELTA (ritorni tra ts consecutivi)
                da = [va[i] - va[i-1] for i in range(1, len(va))]
                db = [vb[i] - vb[i-1] for i in range(1, len(vb))]
                try:
                    corr = statistics.correlation(da, db) if len(da) > 1 else None
                except statistics.StatisticsError:
                    corr = None

    return {"side_by_side": side_by_side, "intraday_pnl_corr": corr}


# ---------- Workstream 2: sensitivity -------------------------------------
SENSITIVITY_DEFAULT_PARAMS: Dict[str, List[str]] = {
    "__global__": ["min_take_edge", "inv_skew_price", "wide_spread_tick", "signal_cap"],
    "ASH_COATED_OSMIUM": ["inv_limit", "size", "hi_widen"],
    "INTARIAN_PEPPER_ROOT": ["w_ofi", "ofi_cap", "pepper_one_sided_gate",
                              "pepper_lean_widen", "inv_widen"],
}


def _grid_for(base_val: float, deltas: Sequence[float]) -> List[float]:
    return [base_val * (1.0 + d) for d in deltas]


def sensitivity_runner(
    backtest_fn: Callable[[Dict[str, Dict[str, Any]]], float],
    deltas: Sequence[float] = (-0.40, -0.20, 0.0, 0.20, 0.40),
    params_scope: Optional[Dict[str, List[str]]] = None,
    red_flag_threshold: float = 0.15,
) -> Dict[str, Any]:
    """WS2 - perturba un parametro alla volta, tutti gli altri fissi.

    backtest_fn(overrides: {product_or_'__global__': {param: new_val}}) -> pnl_totale

    Usa overrides: il backtest_fn deve applicarli via
    `Trader.param_overrides_runtime` (o `trader.attach_overrides`).

    params_scope: se None usa SENSITIVITY_DEFAULT_PARAMS.
    """
    scope = params_scope or SENSITIVITY_DEFAULT_PARAMS
    baseline_pnl = backtest_fn({})

    results: Dict[str, Any] = {"baseline_pnl": baseline_pnl, "params": {}, "red_flags": []}

    for product_key, param_list in scope.items():
        for param in param_list:
            base_val: Optional[float] = None
            if product_key == "__global__":
                if param in BASE_PARAMS:
                    base_val = float(BASE_PARAMS[param])
            else:
                merged = params_for(product_key)
                if param in merged:
                    base_val = float(merged[param])
            if base_val is None:
                continue

            curve: List[Dict[str, float]] = []
            for d in deltas:
                new_val = base_val * (1.0 + d)
                # se parametro intero (size, inv_limit), round
                if param in {"size", "inv_limit", "inv_soft", "wide_spread_tick",
                              "hw_warmup", "liq_min_inv", "liq_chunk",
                              "pepper_impulse_qty"}:
                    new_val = max(1, int(round(new_val)))

                if product_key == "__global__":
                    # applica il param override a TUTTI i prodotti configurati
                    ov = {p: {param: new_val} for p in PARAM_OVERRIDES.keys()}
                    ov.setdefault("__global__", {})[param] = new_val
                else:
                    ov = {product_key: {param: new_val}}

                pnl = backtest_fn(ov)
                curve.append({"delta": d, "value": new_val, "pnl": pnl,
                               "pnl_change_pct": (pnl - baseline_pnl) / abs(baseline_pnl)
                                                   if baseline_pnl != 0 else 0.0})

            # elasticita' locale intorno a delta=0: usa punti -0.2 e +0.2 se presenti
            pt_neg = next((c for c in curve if abs(c["delta"] + 0.2) < 1e-9), None)
            pt_pos = next((c for c in curve if abs(c["delta"] - 0.2) < 1e-9), None)
            elasticity = None
            if pt_neg and pt_pos and baseline_pnl != 0:
                dpnl_pct = (pt_pos["pnl"] - pt_neg["pnl"]) / abs(baseline_pnl)
                dparam_pct = 0.4  # +20% - (-20%)
                elasticity = dpnl_pct / dparam_pct

            sign_flip = False
            if baseline_pnl != 0:
                signs = {(c["pnl"] > 0) for c in curve}
                sign_flip = len(signs) > 1

            best = max(curve, key=lambda c: c["pnl"])

            entry = {
                "base_value": base_val,
                "curve": curve,
                "elasticity_local": elasticity,
                "sign_flip": sign_flip,
                "best_point": best,
            }
            results["params"].setdefault(product_key, {})[param] = entry

            if elasticity is not None and abs(elasticity) > red_flag_threshold:
                results["red_flags"].append({
                    "scope": product_key, "param": param,
                    "elasticity": elasticity,
                })

    return results


# ---------- Workstream 4: OOS robustness ----------------------------------
def session_split_eval(
    fills: Sequence[Fill],
    splits: Sequence[float] = (0.5, 0.7),
    pnl_fn: Callable[[Fill], float] = _fill_pnl_mid,
) -> Dict[str, Any]:
    """WS4-A - split temporale 50/50 e 70/30 sulla stessa sessione.

    NB: NON re-fitta nulla. Calcola solo il PnL delle due meta' con la stessa
    policy, per stimare quanto il risultato sia uniforme nel tempo (se IS e OOS
    sono molto diversi con policy identica -> i parametri attuali sono piu'
    sensibili al regime di quanto sembri).
    """
    sorted_fills = sorted(fills, key=lambda x: (int(x["ts"]), int(x.get("tick", 0))))
    n = len(sorted_fills)
    out: Dict[str, Any] = {}
    for s in splits:
        k = int(n * s)
        first = sorted_fills[:k]
        second = sorted_fills[k:]
        pnl_first = sum(pnl_fn(f) for f in first)
        pnl_second = sum(pnl_fn(f) for f in second)
        rate_first = pnl_first / k if k else 0.0
        rate_second = pnl_second / (n - k) if (n - k) else 0.0
        degrade = (rate_second - rate_first) / abs(rate_first) if rate_first else None
        out[f"split_{int(s*100)}_{100-int(s*100)}"] = {
            "n_first": k, "n_second": n - k,
            "pnl_first": pnl_first, "pnl_second": pnl_second,
            "rate_first_per_fill": rate_first,
            "rate_second_per_fill": rate_second,
            "degrade_rate": degrade,
        }
    return out


def walk_forward_stats(
    fills: Sequence[Fill],
    window_ts: int = 50_000,
    step_ts: int = 25_000,
    pnl_fn: Callable[[Fill], float] = _fill_pnl_mid,
) -> List[Dict[str, Any]]:
    """WS4-B - finestre walk-forward su timestamp.

    Ritorna metriche per finestra: pnl totale, n_fills, hit_rate, moments.
    Stabilita' si vede plottando la serie.
    """
    if not fills:
        return []
    sorted_fills = sorted(fills, key=lambda x: (int(x["ts"]), int(x.get("tick", 0))))
    ts_min = int(sorted_fills[0]["ts"])
    ts_max = int(sorted_fills[-1]["ts"])

    out: List[Dict[str, Any]] = []
    start = ts_min
    while start <= ts_max:
        end = start + window_ts
        w = [f for f in sorted_fills if start <= int(f["ts"]) < end]
        if w:
            pnls = [pnl_fn(f) for f in w]
            pnl_sum = sum(pnls)
            hit = sum(1 for x in pnls if x > 0) / len(pnls)
            moms = _moments(pnls)
            out.append({
                "window_start": start, "window_end": end,
                "n_fills": len(w), "pnl": pnl_sum,
                "hit_rate": hit, "moments": moms,
            })
        start += step_ts
    return out


def block_bootstrap(
    fills: Sequence[Fill],
    block_size: int = 500,
    n_bootstrap: int = 1000,
    seed: int = 42,
    pnl_fn: Callable[[Fill], float] = _fill_pnl_mid,
) -> Dict[str, float]:
    """WS4-C - bootstrap a blocchi per intervallo di confidenza su PnL e Sharpe.

    Usa block resampling per preservare autocorrelazione dei trade.
    Ritorna mediana e percentili 5/95 di PnL totale e Sharpe naive
    (mean/std * sqrt(n) sulla sequenza di trade PnL).
    """
    if not fills:
        return {"median_pnl": 0.0, "p05_pnl": 0.0, "p95_pnl": 0.0,
                "median_sharpe": 0.0, "p05_sharpe": 0.0, "p95_sharpe": 0.0,
                "prob_edge_positive": 0.0}

    sorted_fills = sorted(fills, key=lambda x: (int(x["ts"]), int(x.get("tick", 0))))
    pnls = [pnl_fn(f) for f in sorted_fills]
    n = len(pnls)
    block_size = max(1, min(block_size, n))

    rnd = random.Random(seed)
    n_blocks = max(1, n // block_size)

    boot_pnl: List[float] = []
    boot_sharpe: List[float] = []
    for _ in range(n_bootstrap):
        sample: List[float] = []
        for _b in range(n_blocks):
            start = rnd.randint(0, n - block_size)
            sample.extend(pnls[start:start + block_size])
        total = sum(sample)
        boot_pnl.append(total)
        if len(sample) > 1:
            m = sum(sample) / len(sample)
            var = sum((x - m) ** 2 for x in sample) / (len(sample) - 1)
            sd = math.sqrt(var)
            sharpe = (m / sd) * math.sqrt(len(sample)) if sd > 0 else 0.0
        else:
            sharpe = 0.0
        boot_sharpe.append(sharpe)

    def _q(xs: Sequence[float], p: float) -> float:
        s = sorted(xs)
        idx = int(p * (len(s) - 1))
        return s[idx]

    return {
        "median_pnl": statistics.median(boot_pnl),
        "p05_pnl": _q(boot_pnl, 0.05),
        "p95_pnl": _q(boot_pnl, 0.95),
        "median_sharpe": statistics.median(boot_sharpe),
        "p05_sharpe": _q(boot_sharpe, 0.05),
        "p95_sharpe": _q(boot_sharpe, 0.95),
        "prob_edge_positive": sum(1 for x in boot_pnl if x > 0) / len(boot_pnl),
    }


# ===========================================================================
# Done-criteria helper: aggrega le risposte ai 4 quesiti del prompt.
# ===========================================================================
def summarize_done_criteria(
    attribution: Dict[str, Any],
    per_product: Dict[str, Dict[str, Any]],
    sensitivity: Dict[str, Any],
    oos: Dict[str, Any],
) -> Dict[str, Any]:
    # (1) miglior blocco per PnL/fill
    best_block = None
    for tag, r in attribution["by_tag"].items():
        if r["fills"] > 0:
            if best_block is None or r["pnl_per_fill"] > best_block[1]:
                best_block = (tag, r["pnl_per_fill"])

    # (2) parametro piu' fragile (|elasticity| max)
    most_fragile = None
    for scope, pmap in sensitivity.get("params", {}).items():
        for param, entry in pmap.items():
            el = entry.get("elasticity_local")
            if el is None:
                continue
            if most_fragile is None or abs(el) > abs(most_fragile[2]):
                most_fragile = (scope, param, el)

    # (3) dipendenza COAT vs PEPPER (usa compare_products sulla curva cumulata)
    corr = None
    if len(per_product) == 2:
        corr = compare_products(per_product).get("intraday_pnl_corr")

    # (4) degrado OOS (usa splits se disponibili)
    degrade = None
    for key, s in oos.get("splits", {}).items():
        d = s.get("degrade_rate")
        if d is None:
            continue
        if degrade is None or abs(d) > abs(degrade[1]):
            degrade = (key, d)

    return {
        "best_block_pnl_per_fill": best_block,
        "most_fragile_param": most_fragile,
        "coat_pepper_pnl_corr": corr,
        "worst_oos_degrade": degrade,
    }


__all__ = [
    "BASE_PARAMS", "PARAM_OVERRIDES", "DEFAULT_UNKNOWN_PARAMS", "params_for",
    "HullWhiteSV", "OFIState",
    "Trader", "SymbolState",
    "Diagnostics", "TaggedOrder", "TickSnapshot",
    "attribute_pnl_by_tag", "per_product_stats", "compare_products",
    "sensitivity_runner", "session_split_eval",
    "walk_forward_stats", "block_bootstrap",
    "summarize_done_criteria",
]
