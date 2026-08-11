"""Reusable epistemic and conformal collision-risk estimators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import PocketWorldModel


class PocketWorldEnsemble:
    """Aggregate collision forecasts from independently trained checkpoints.

    The wrapper deliberately exposes only the collision-risk interface used by
    the planners. Position rollouts are averaged for callers that want a
    direct ensemble rollout, while ordinary planner comparisons can keep the
    primary dynamics model and inject this object as ``collision_model``.
    """

    def __init__(self, members: list[PocketWorldModel], disagreement_weight: float = 1.0) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        if disagreement_weight < 0:
            raise ValueError("disagreement_weight must be non-negative")
        self.members = members
        self.disagreement_weight = float(disagreement_weight)
        self.last_statistics: dict[str, float] = {}

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: list[str | Path],
        disagreement_weight: float = 1.0,
    ) -> "PocketWorldEnsemble":
        members: list[PocketWorldModel] = []
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location="cpu")
            member = PocketWorldModel()
            member.load_state_dict(payload["model"], strict=False)
            members.append(member)
        return cls(members, disagreement_weight=disagreement_weight)

    def eval(self) -> "PocketWorldEnsemble":
        for member in self.members:
            member.eval()
        return self

    def __getattr__(self, name: str) -> Any:
        # The first member supplies scalar dynamics parameters and helper
        # methods such as encode/temporal_velocity_stats. Risk-specific calls
        # below are intentionally overridden instead of silently delegated.
        members = self.__dict__.get("members")
        if members:
            return getattr(members[0], name)
        raise AttributeError(name)

    @torch.no_grad()
    def imagine_positions(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        predictions = [member.imagine_positions(*args, **kwargs) for member in self.members]
        return torch.stack(predictions, dim=0).mean(dim=0)

    @torch.no_grad()
    def imagine_collision_probabilities(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        predictions = [member.imagine_collision_probabilities(*args, **kwargs) for member in self.members]
        stacked = torch.stack(predictions, dim=0)
        mean = stacked.mean(dim=0)
        disagreement = stacked.std(dim=0, unbiased=False)
        upper_risk = (mean + self.disagreement_weight * disagreement).clamp(0.0, 1.0)
        self.last_statistics = {
            "mean_risk": float(mean.mean().item()),
            "disagreement_std": float(disagreement.mean().item()),
            "upper_risk": float(upper_risk.mean().item()),
        }
        return upper_risk


@dataclass(frozen=True)
class ConformalRiskCalibration:
    """Finite-sample split-conformal upper-risk calibration."""

    alpha: float
    quantile: float
    samples: int
    collision_rate: float


def fit_conformal_upper_bound(
    predicted_risk: np.ndarray,
    collision_labels: np.ndarray,
    alpha: float = 0.10,
) -> ConformalRiskCalibration:
    """Fit a conservative upper bound for binary collision events.

    The calibration score is ``label - predicted_risk``. The finite-sample
    order statistic gives an upper bound whose empirical coverage can be
    checked on held-out routes. The quantile is clipped at zero so calibration
    never turns a risk estimate into an unjustifiably smaller value.
    """
    risks = np.asarray(predicted_risk, dtype=np.float64).reshape(-1)
    labels = np.asarray(collision_labels, dtype=np.float64).reshape(-1)
    if risks.size == 0 or risks.size != labels.size:
        raise ValueError("predicted_risk and collision_labels must be non-empty and equally sized")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not np.isfinite(risks).all() or not np.isfinite(labels).all():
        raise ValueError("calibration arrays must be finite")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("collision_labels must be binary")
    scores = labels - np.clip(risks, 0.0, 1.0)
    rank = int(np.ceil((scores.size + 1) * (1.0 - alpha))) - 1
    rank = int(np.clip(rank, 0, scores.size - 1))
    quantile = float(max(0.0, np.sort(scores)[rank]))
    return ConformalRiskCalibration(
        alpha=float(alpha),
        quantile=quantile,
        samples=int(scores.size),
        collision_rate=float(labels.mean()),
    )


class ConformalCollisionRisk:
    """Add a split-conformal upper margin to another risk estimator."""

    def __init__(self, base: Any, calibration: ConformalRiskCalibration) -> None:
        self.base = base
        self.calibration = calibration

    def eval(self) -> "ConformalCollisionRisk":
        self.base.eval()
        return self

    @torch.no_grad()
    def imagine_collision_probabilities(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        risk = self.base.imagine_collision_probabilities(*args, **kwargs)
        return (risk + self.calibration.quantile).clamp(0.0, 1.0)

