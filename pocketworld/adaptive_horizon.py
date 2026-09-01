"""Transparent, hysteretic selection of a model-based planning horizon.

This module deliberately contains no simulator access.  A caller supplies
online-observable diagnostics (calibrated transition uncertainty, learned
collision risk, route alignment, shift score, and recent planning pressure),
and the policy returns an auditable horizon decision.  The distinction from
``adaptive_mpc_decision`` is intentional: that older policy selects an
ordinary or robust solver at one fixed horizon; this policy selects the number
of imagined steps before the solver is called.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_HORIZONS = (8, 16, 24, 32)


def _finite_unit(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(value, 0.0, 1.0))


def _normalise_horizon_values(
    value: float | Mapping[int, float], horizons: tuple[int, ...], name: str
) -> dict[int, float]:
    if isinstance(value, Mapping):
        missing = [horizon for horizon in horizons if horizon not in value]
        if missing:
            raise ValueError(f"{name} is missing horizons: {missing}")
        return {horizon: _finite_unit(value[horizon], f"{name}[{horizon}]") for horizon in horizons}
    scalar = _finite_unit(value, name)
    return {horizon: scalar for horizon in horizons}


@dataclass(frozen=True)
class HorizonDecision:
    """One online horizon choice and the evidence used to make it."""

    horizon: int
    uncertainty_score: float
    collision_risk: float
    alignment_error: float
    reason: str
    ood_score: float = 0.0
    recent_risk: float = 0.0
    risk_score: float = 0.0
    risk_budget: float = 0.0
    previous_horizon: int | None = None
    switched: bool = False
    candidate_risks: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe decision fields, including every candidate risk."""
        payload = asdict(self)
        payload["candidate_risks"] = {
            str(int(key)): float(value) for key, value in self.candidate_risks.items()
        }
        return payload


class AdaptiveHorizonPolicy:
    """Choose the longest risk-feasible horizon with hysteresis.

    The transparent score for candidate horizon ``h`` is:

    ``R_h = 0.40 U_h + 0.30 C_h + 0.15 A + 0.10 O + 0.05 P``

    where ``U_h`` is calibrated cumulative position uncertainty, ``C_h`` is
    calibrated cumulative collision risk, ``A`` is current route alignment
    error divided by ``alignment_budget_px``, ``O`` is the label-free shift
    score divided by ``ood_budget``, and ``P`` is recent planning pressure.
    Each term is clipped to ``[0, 1]``.  The policy selects the longest
    candidate with ``R_h <= risk_budget``.  If none is feasible it returns the
    shortest candidate.  Between ``exit_threshold`` and ``entry_threshold`` it
    holds the previous horizon when possible, preventing rapid oscillation.
    """

    def __init__(
        self,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        risk_budget: float = 0.45,
        entry_threshold: float = 0.55,
        exit_threshold: float = 0.35,
        alignment_budget_px: float = 6.0,
        ood_budget: float = 2.0,
    ) -> None:
        values = tuple(sorted({int(horizon) for horizon in horizons}))
        if not values or any(horizon < 1 for horizon in values):
            raise ValueError("horizons must contain positive values")
        if not 0.0 < risk_budget <= 1.0:
            raise ValueError("risk_budget must be in (0, 1]")
        if not 0.0 <= exit_threshold < entry_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= exit < entry <= 1")
        if not np.isfinite(alignment_budget_px) or alignment_budget_px <= 0.0:
            raise ValueError("alignment_budget_px must be finite and positive")
        if not np.isfinite(ood_budget) or ood_budget <= 0.0:
            raise ValueError("ood_budget must be finite and positive")
        self.horizons = values
        self.risk_budget = float(risk_budget)
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.alignment_budget_px = float(alignment_budget_px)
        self.ood_budget = float(ood_budget)

    def risk_score(
        self,
        uncertainty_score: float,
        collision_risk: float,
        alignment_error: float,
        ood_score: float = 0.0,
        recent_risk: float = 0.0,
    ) -> float:
        """Compute the documented scalar risk score from online signals."""
        uncertainty = _finite_unit(uncertainty_score, "uncertainty_score")
        collision = _finite_unit(collision_risk, "collision_risk")
        alignment = _finite_unit(float(alignment_error) / self.alignment_budget_px, "alignment_error")
        ood = _finite_unit(float(ood_score) / self.ood_budget, "ood_score")
        recent = _finite_unit(recent_risk, "recent_risk")
        return float(
            np.clip(
                0.40 * uncertainty
                + 0.30 * collision
                + 0.15 * alignment
                + 0.10 * ood
                + 0.05 * recent,
                0.0,
                1.0,
            )
        )

    def select_horizon(
        self,
        uncertainty_score: float | Mapping[int, float],
        collision_risk: float | Mapping[int, float],
        alignment_error: float,
        ood_score: float = 0.0,
        recent_risk: float = 0.0,
        previous_horizon: int | None = None,
    ) -> HorizonDecision:
        """Select a horizon using only the supplied online-observable values.

        ``uncertainty_score`` and ``collision_risk`` may be scalars or maps
        keyed by candidate horizon.  Mapping values let the evaluator expose
        the cumulative risk curve instead of hiding it inside a learned gate.
        """
        if previous_horizon is not None and int(previous_horizon) not in self.horizons:
            raise ValueError("previous_horizon must be one of the candidate horizons")
        alignment_error = float(alignment_error)
        if not np.isfinite(alignment_error) or alignment_error < 0.0:
            raise ValueError("alignment_error must be finite and non-negative")
        uncertainty = _normalise_horizon_values(uncertainty_score, self.horizons, "uncertainty_score")
        collision = _normalise_horizon_values(collision_risk, self.horizons, "collision_risk")
        candidate_risks = {
            horizon: self.risk_score(
                uncertainty[horizon],
                collision[horizon],
                alignment_error,
                ood_score,
                recent_risk,
            )
            for horizon in self.horizons
        }
        feasible = [horizon for horizon in self.horizons if candidate_risks[horizon] <= self.risk_budget]
        shortest = self.horizons[0]
        previous = int(previous_horizon) if previous_horizon is not None else None
        if not feasible:
            selected = shortest
            reason = "all_candidates_exceed_risk_budget"
        elif previous is None:
            selected = max(feasible)
            reason = "longest_risk_feasible_initial"
        else:
            previous_risk = candidate_risks[previous]
            if previous_risk >= self.entry_threshold:
                shorter_feasible = [horizon for horizon in feasible if horizon <= previous]
                selected = max(shorter_feasible) if shorter_feasible else shortest
                reason = "entry_threshold_shorten" if selected < previous else "entry_threshold_hold"
            elif previous_risk <= self.exit_threshold:
                selected = max(feasible)
                reason = "exit_threshold_extend" if selected > previous else "exit_threshold_hold"
            elif previous in feasible:
                selected = previous
                reason = "hysteresis_hold"
            else:
                shorter_feasible = [horizon for horizon in feasible if horizon <= previous]
                selected = max(shorter_feasible) if shorter_feasible else shortest
                reason = "hysteresis_budget_shorten"
        selected_uncertainty = uncertainty[selected]
        selected_collision = collision[selected]
        selected_risk = candidate_risks[selected]
        return HorizonDecision(
            horizon=int(selected),
            uncertainty_score=float(selected_uncertainty),
            collision_risk=float(selected_collision),
            alignment_error=alignment_error,
            ood_score=_finite_unit(ood_score / self.ood_budget, "ood_score") * self.ood_budget,
            recent_risk=_finite_unit(recent_risk, "recent_risk"),
            risk_score=float(selected_risk),
            risk_budget=self.risk_budget,
            previous_horizon=previous,
            switched=previous is not None and int(selected) != previous,
            reason=reason,
            candidate_risks=candidate_risks,
        )


def validate_horizon_candidates(horizons: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Validate and canonicalise a CLI horizon list."""
    policy = AdaptiveHorizonPolicy(tuple(horizons))
    return policy.horizons

