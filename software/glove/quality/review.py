"""Read/write the human-review block in `meta.yaml`.

The dashboard writes one of {approve, reject, force_include} into
`review.decision` and stamps `review.decided_at`. This module is the only
place that mutates `meta.yaml['review']` so the schema stays consistent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from glove.quality.schema import ReviewBlock, ReviewDecision


def load_review(session_dir: Path) -> ReviewBlock:
    meta_path = session_dir / "meta.yaml"
    if not meta_path.exists():
        return ReviewBlock()
    meta = yaml.safe_load(meta_path.read_text()) or {}
    return ReviewBlock.from_dict(meta.get("review"))


def save_review(
    session_dir: Path,
    decision: Optional[ReviewDecision],
    notes: Optional[str] = None,
) -> ReviewBlock:
    meta_path = session_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.yaml in {session_dir}")

    meta = yaml.safe_load(meta_path.read_text()) or {}
    existing = meta.get("review") or {}

    block = ReviewBlock(
        decision=decision,
        decided_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=notes if notes is not None else existing.get("notes", ""),
    )
    meta["review"] = block.to_dict()
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return block
