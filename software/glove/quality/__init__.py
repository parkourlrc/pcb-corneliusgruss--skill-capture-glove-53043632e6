"""Demo quality scoring and verdict logic.

Defines the per-demo quality metric schema, verdict thresholds, and dataset
manifest filtering. Metrics are computed by `glove.quality.compute` from
`aligned/episode.parquet`; verdicts (CLEAN / NOISY / RAILS) gate inclusion
in training manifests.
"""

from glove.quality.schema import (
    QualityMetrics,
    QualityBlock,
    ReviewDecision,
    Verdict,
    compute_verdict,
    THRESHOLDS,
)

__all__ = [
    "QualityMetrics",
    "QualityBlock",
    "ReviewDecision",
    "Verdict",
    "compute_verdict",
    "THRESHOLDS",
]
