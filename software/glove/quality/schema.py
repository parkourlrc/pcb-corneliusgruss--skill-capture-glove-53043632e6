"""Quality schema — the contract between `align` and the review dashboard.

`meta.yaml` gains two top-level blocks once a session has been scored + reviewed:

    quality:
      metrics:
        pose_jump_max_mm: 47.2
        pose_jump_count_20mm: 3
        pose_vel_p95_mps: 0.42
        pose_accel_p95: 1.8
        quat_angle_jump_max_deg: 3.1
        relocalization_suspects: 2
        lost_ratio: 0.009
        lost_run_max: 2
        proprio_pose_coherence: 0.81
        duration_s: 22.1
        n_frames: 1326
        n_keyframes: 54
      verdict: needs_review          # auto_pass | auto_fail | needs_review
      reasons:
        - "pose_jump_max_mm=47.2 exceeds 20.0 threshold"
        - "relocalization_suspects=2 exceeds 1 threshold"
      flagged_frames: [1247, 1251]
      scored_at: "2026-04-23T18:00:12Z"
      schema_version: 1

    review:
      decision: approve              # approve | reject | force_include | null
      decided_at: "2026-04-23T18:01:04Z"
      notes: ""

`verdict` is the auto-gate output. `review.decision` is the human override
(or concurrence). Training manifest filtering uses `review.decision` if
present, else `verdict` (auto_pass included, auto_fail excluded,
needs_review excluded until reviewed).
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, asdict
from datetime import datetime, timezone
from typing import Literal, Optional

# v2 adds local_straightness_0_25s + per_axis_hifreq_max_mm to capture sub-jump
# wobble that per-frame jump thresholds normalize away. Threshold values were
# tuned against a known-clean reference demo, not against ground truth.
SCHEMA_VERSION = 2

Verdict = Literal["auto_pass", "auto_fail", "needs_review"]
ReviewDecision = Literal["approve", "reject", "force_include"]


@dataclass(frozen=True)
class Thresholds:
    """Verdict thresholds. Tune these as the data tells us more."""

    # Hard-fail (any one triggers auto_fail)
    pose_jump_max_mm_fail: float = 50.0
    pose_vel_p95_mps_fail: float = 1.2
    relocalization_suspects_fail: int = 2

    # Soft-warn (any one triggers needs_review; fail thresholds above are stricter)
    pose_jump_max_mm_warn: float = 20.0
    pose_vel_p95_mps_warn: float = 0.6
    quat_angle_jump_max_deg_warn: float = 10.0
    relocalization_suspects_warn: int = 1
    lost_ratio_warn: float = 0.05
    lost_run_max_warn: int = 5
    # Wobble gates (v2). Tuned against a known-clean reference demo
    # (local_straightness ≈ 0.85, per-axis hifreq RMS 0.4–0.8 mm). Pose-graph
    # re-anchoring artifacts typically score 0.68–0.76 / 1.0–2.2 mm and are
    # flagged as needs_review by these gates. The per-frame jump max (≤ 20 mm)
    # is too coarse to catch back-and-forth sub-jump wobble that bricks policy
    # training — hence the additional shape and high-frequency gates.
    local_straightness_0_25s_warn: float = 0.80
    per_axis_hifreq_max_mm_warn: float = 1.0
    # proprio_pose_coherence is tracked but NOT a gate input — its current
    # rolling-correlation definition is noisy and often near zero for normal
    # demos. Refine before wiring it into the verdict.
    duration_s_min: float = 3.0
    duration_s_max: float = 120.0


THRESHOLDS = Thresholds()


@dataclass
class QualityMetrics:
    pose_jump_max_mm: float
    pose_jump_count_20mm: int
    pose_vel_p95_mps: float
    pose_accel_p95: float
    quat_angle_jump_max_deg: float
    relocalization_suspects: int
    lost_ratio: float
    lost_run_max: int
    proprio_pose_coherence: float
    duration_s: float
    n_frames: int
    n_keyframes: int
    # v2 wobble gates — see Thresholds for rationale.
    # Defaults are "perfect" so that older meta.yaml blocks (schema_version=1)
    # round-trip cleanly through from_dict() and never auto-trigger the new
    # warns; re-score to populate real values.
    local_straightness_0_25s: float = 1.0
    per_axis_hifreq_max_mm: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "QualityMetrics":
        kwargs = {}
        for name, fld in cls.__dataclass_fields__.items():
            if name in d:
                kwargs[name] = d[name]
            elif fld.default is not MISSING:
                kwargs[name] = fld.default
            elif fld.default_factory is not MISSING:  # pragma: no cover
                kwargs[name] = fld.default_factory()
            else:
                raise KeyError(f"QualityMetrics missing required field {name!r}")
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityBlock:
    metrics: QualityMetrics
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    flagged_frames: list[int] = field(default_factory=list)
    scored_at: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, d: dict) -> "QualityBlock":
        return cls(
            metrics=QualityMetrics.from_dict(d["metrics"]),
            verdict=d.get("verdict", "needs_review"),
            reasons=list(d.get("reasons", [])),
            flagged_frames=list(d.get("flagged_frames", [])),
            scored_at=d.get("scored_at", ""),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "flagged_frames": list(self.flagged_frames),
            "scored_at": self.scored_at,
            "schema_version": self.schema_version,
        }


def compute_verdict(
    metrics: QualityMetrics,
    thresholds: Thresholds = THRESHOLDS,
    flagged_frames: Optional[list[int]] = None,
) -> QualityBlock:
    """Classify a demo based on its metrics.

    Returns a QualityBlock with `verdict`, `reasons`, and `flagged_frames`
    populated. `flagged_frames` is filled upstream by `glove.quality.compute`
    (the frame indices where pose jumps triggered); pass through unchanged.
    """
    t = thresholds
    reasons_fail: list[str] = []
    reasons_warn: list[str] = []

    # Hard-fail checks
    if metrics.pose_jump_max_mm >= t.pose_jump_max_mm_fail:
        reasons_fail.append(
            f"pose_jump_max_mm={metrics.pose_jump_max_mm:.1f} ≥ {t.pose_jump_max_mm_fail} (fail)"
        )
    if metrics.pose_vel_p95_mps >= t.pose_vel_p95_mps_fail:
        reasons_fail.append(
            f"pose_vel_p95_mps={metrics.pose_vel_p95_mps:.2f} ≥ {t.pose_vel_p95_mps_fail} (fail)"
        )
    if metrics.relocalization_suspects >= t.relocalization_suspects_fail:
        reasons_fail.append(
            f"relocalization_suspects={metrics.relocalization_suspects} ≥ {t.relocalization_suspects_fail} (fail)"
        )

    # Soft-warn checks
    if metrics.pose_jump_max_mm >= t.pose_jump_max_mm_warn:
        reasons_warn.append(
            f"pose_jump_max_mm={metrics.pose_jump_max_mm:.1f} ≥ {t.pose_jump_max_mm_warn}"
        )
    if metrics.pose_vel_p95_mps >= t.pose_vel_p95_mps_warn:
        reasons_warn.append(
            f"pose_vel_p95_mps={metrics.pose_vel_p95_mps:.2f} ≥ {t.pose_vel_p95_mps_warn}"
        )
    if metrics.quat_angle_jump_max_deg >= t.quat_angle_jump_max_deg_warn:
        reasons_warn.append(
            f"quat_angle_jump_max_deg={metrics.quat_angle_jump_max_deg:.1f} ≥ {t.quat_angle_jump_max_deg_warn}"
        )
    if metrics.relocalization_suspects >= t.relocalization_suspects_warn:
        reasons_warn.append(
            f"relocalization_suspects={metrics.relocalization_suspects} ≥ {t.relocalization_suspects_warn}"
        )
    if metrics.lost_ratio >= t.lost_ratio_warn:
        reasons_warn.append(f"lost_ratio={metrics.lost_ratio:.1%} ≥ {t.lost_ratio_warn:.1%}")
    if metrics.lost_run_max >= t.lost_run_max_warn:
        reasons_warn.append(f"lost_run_max={metrics.lost_run_max} ≥ {t.lost_run_max_warn}")
    # Wobble gates — note straightness is a *minimum* (lower = worse), hifreq
    # is a *maximum* (higher = noisier).
    if metrics.local_straightness_0_25s < t.local_straightness_0_25s_warn:
        reasons_warn.append(
            f"local_straightness_0_25s={metrics.local_straightness_0_25s:.3f} < "
            f"{t.local_straightness_0_25s_warn:.2f}"
        )
    if metrics.per_axis_hifreq_max_mm >= t.per_axis_hifreq_max_mm_warn:
        reasons_warn.append(
            f"per_axis_hifreq_max_mm={metrics.per_axis_hifreq_max_mm:.2f} ≥ "
            f"{t.per_axis_hifreq_max_mm_warn:.2f}"
        )
    if metrics.duration_s < t.duration_s_min:
        reasons_warn.append(f"duration_s={metrics.duration_s:.1f} < {t.duration_s_min}")
    if metrics.duration_s > t.duration_s_max:
        reasons_warn.append(f"duration_s={metrics.duration_s:.1f} > {t.duration_s_max}")

    if reasons_fail:
        verdict: Verdict = "auto_fail"
        reasons = reasons_fail
    elif reasons_warn:
        verdict = "needs_review"
        reasons = reasons_warn
    else:
        verdict = "auto_pass"
        reasons = []

    return QualityBlock(
        metrics=metrics,
        verdict=verdict,
        reasons=reasons,
        flagged_frames=list(flagged_frames or []),
        scored_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@dataclass
class ReviewBlock:
    decision: Optional[ReviewDecision] = None
    decided_at: Optional[str] = None
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ReviewBlock":
        if not d:
            return cls()
        return cls(
            decision=d.get("decision"),
            decided_at=d.get("decided_at"),
            notes=d.get("notes", ""),
        )

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "decided_at": self.decided_at,
            "notes": self.notes,
        }


def effective_inclusion(
    quality: Optional[QualityBlock], review: Optional[ReviewBlock]
) -> bool:
    """Should this demo be included in the training manifest?

    Rules:
    - Human `reject` always excludes (overrides auto_pass).
    - Human `force_include` always includes (overrides auto_fail).
    - Human `approve` includes (validates auto_pass or flips needs_review).
    - No human decision:
        - auto_pass → include
        - auto_fail → exclude
        - needs_review → exclude (not yet reviewed)
    - No quality block → exclude (not yet scored).
    """
    if review and review.decision == "reject":
        return False
    if review and review.decision == "force_include":
        return True
    if review and review.decision == "approve":
        return True
    if quality is None:
        return False
    return quality.verdict == "auto_pass"
