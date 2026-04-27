"""Session scanner for the review UI.

Reads new-format (`meta.yaml` + `aligned/episode.parquet`) sessions and
surfaces the summary info the review queue needs. Separate from the old
dashboard session loader, which still serves the capture/review surfaces
on the legacy `session_*` format.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional

import yaml

from glove.quality.schema import (
    QualityBlock,
    ReviewBlock,
    effective_inclusion,
)

QueueFilter = Literal["needs_review", "auto_fail", "auto_pass", "decided", "all"]


@dataclass
class ReviewSession:
    session_id: str
    path: str
    task: str
    kind: str            # "demo" | "mapping"
    ordinal: Optional[int]
    created_at: str
    duration_s: float
    has_episode: bool
    quality: Optional[dict]        # demo quality block, or None
    review: Optional[dict]         # demo review block, or None
    map_quality: Optional[dict] = None   # mapping sessions only
    scene: Optional[str] = None          # mapping sessions only
    parent_map: Optional[str] = None     # demo → which map it localized against

    @property
    def is_map(self) -> bool:
        return self.kind == "mapping"

    @property
    def is_demo(self) -> bool:
        return self.kind == "demo"

    @property
    def verdict(self) -> str:
        return (self.quality or {}).get("verdict", "unscored")

    @property
    def decision(self) -> Optional[str]:
        return (self.review or {}).get("decision")

    @property
    def reviewed(self) -> bool:
        return self.decision is not None

    @property
    def included_in_manifest(self) -> bool:
        q = QualityBlock.from_dict(self.quality) if self.quality else None
        r = ReviewBlock.from_dict(self.review) if self.review else ReviewBlock()
        return effective_inclusion(q, r)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_session(session_dir: Path) -> Optional[ReviewSession]:
    meta_path = session_dir / "meta.yaml"
    if not meta_path.exists():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except Exception:
        return None

    kind = meta.get("kind", "demo")
    if kind not in ("demo", "mapping"):
        return None

    episode_path = session_dir / "aligned" / "episode.parquet"
    video_meta = meta.get("video", {}) or {}

    return ReviewSession(
        session_id=meta.get("session_id", session_dir.name),
        path=str(session_dir),
        task=meta.get("task", ""),
        kind=kind,
        ordinal=meta.get("ordinal"),
        created_at=str(meta.get("created_at", "")),
        duration_s=float(video_meta.get("duration_s", 0.0) or 0.0),
        has_episode=episode_path.exists(),
        quality=meta.get("quality"),
        review=meta.get("review"),
        map_quality=meta.get("map_quality"),
        scene=meta.get("scene"),
        parent_map=meta.get("parent_map"),
    )


def _scan_all(data_dir: Path) -> list[ReviewSession]:
    results: list[ReviewSession] = []
    if not data_dir.exists():
        return results
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue
        if not (p.name.startswith("demo_") or p.name.startswith("map_")):
            continue
        s = _load_session(p)
        if s is not None:
            results.append(s)
    results.sort(key=lambda s: (s.created_at, s.session_id))
    return results


def scan_sessions(data_dir: Path) -> list[ReviewSession]:
    """Demo sessions only, oldest-first (FIFO for review queue)."""
    return [s for s in _scan_all(data_dir) if s.is_demo]


def scan_maps(data_dir: Path) -> list[ReviewSession]:
    """Mapping sessions only, newest-first."""
    maps = [s for s in _scan_all(data_dir) if s.is_map]
    maps.sort(key=lambda s: (s.created_at, s.session_id), reverse=True)
    return maps


def demos_by_map(demos: list[ReviewSession]) -> dict[str, list[ReviewSession]]:
    """Group demos by their `parent_map` field. Demos without one land under `''`."""
    buckets: dict[str, list[ReviewSession]] = {}
    for d in demos:
        buckets.setdefault(d.parent_map or "", []).append(d)
    return buckets


def filter_queue(
    sessions: list[ReviewSession],
    queue: QueueFilter,
    task: Optional[str] = None,
) -> list[ReviewSession]:
    out = []
    for s in sessions:
        if task and s.task != task:
            continue
        if queue == "needs_review":
            # Only unreviewed needs_review sessions appear in the default queue
            if s.verdict != "needs_review" or s.reviewed:
                continue
        elif queue == "auto_fail":
            if s.verdict != "auto_fail":
                continue
        elif queue == "auto_pass":
            if s.verdict != "auto_pass":
                continue
        elif queue == "decided":
            if not s.reviewed:
                continue
        # queue == "all" → no verdict filter
        out.append(s)
    return out


def summary_counts(sessions: list[ReviewSession]) -> dict:
    total = len(sessions)
    auto_pass = sum(1 for s in sessions if s.verdict == "auto_pass")
    auto_fail = sum(1 for s in sessions if s.verdict == "auto_fail")
    needs_review = sum(1 for s in sessions if s.verdict == "needs_review")
    unscored = sum(1 for s in sessions if s.verdict == "unscored")
    reviewed = sum(1 for s in sessions if s.reviewed)
    pending_review = sum(
        1 for s in sessions if s.verdict == "needs_review" and not s.reviewed
    )
    included = sum(1 for s in sessions if s.included_in_manifest)
    tasks = sorted({s.task for s in sessions if s.task})
    return {
        "total": total,
        "auto_pass": auto_pass,
        "auto_fail": auto_fail,
        "needs_review": needs_review,
        "unscored": unscored,
        "reviewed": reviewed,
        "pending_review": pending_review,
        "included": included,
        "tasks": tasks,
    }


def find_session(sessions: list[ReviewSession], session_id: str) -> Optional[ReviewSession]:
    for s in sessions:
        if s.session_id == session_id:
            return s
    return None


def neighbor_ids(
    sessions: list[ReviewSession], current_id: str
) -> tuple[Optional[str], Optional[str]]:
    """Return (prev_id, next_id) within the provided (filtered) list."""
    ids = [s.session_id for s in sessions]
    if current_id not in ids:
        return (None, None)
    i = ids.index(current_id)
    prev_id = ids[i - 1] if i > 0 else None
    next_id = ids[i + 1] if i + 1 < len(ids) else None
    return (prev_id, next_id)
