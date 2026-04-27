"""Review routes — the queue list and per-session inspection workstation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from glove.dashboard import scrub_cache
from glove.dashboard.episode_panel import load_episode_panel
from glove.dashboard.review_queue import (
    filter_queue,
    find_session,
    neighbor_ids,
    scan_sessions,
    summary_counts,
)
from glove.dashboard.frame_extract import frame_from_mp4
from glove.dashboard.templating import make_templates
from glove.quality.review import save_review

router = APIRouter()
templates = make_templates()


def _data_dir() -> Path:
    # Pulled lazily so tests can swap it via dashboard.app.data_dir
    from glove.dashboard import app as _app
    return _app.data_dir


@router.get("/review", response_class=HTMLResponse)
async def review_index(
    request: Request,
    queue: str = Query("all"),
    task: Optional[str] = Query(None),
    sort: str = Query("created"),
    order: str = Query("asc"),
    q: Optional[str] = Query(None),
):
    """Queue list across all sessions, filtered by verdict + task."""
    sessions = scan_sessions(_data_dir())
    counts = summary_counts(sessions)
    filtered = filter_queue(sessions, queue=queue, task=task)

    if q:
        ql = q.lower()
        filtered = [
            s for s in filtered
            if ql in s.session_id.lower() or ql in (s.task or "").lower()
        ]

    def _sort_key(s):
        if sort == "jump_max":
            return (s.quality or {}).get("metrics", {}).get("pose_jump_max_mm", 0.0)
        if sort == "duration":
            return s.duration_s
        if sort == "task":
            return s.task or ""
        if sort == "session_id":
            return s.session_id
        if sort == "verdict":
            verdict_rank = {"auto_fail": 0, "needs_review": 1, "unscored": 2, "auto_pass": 3}
            return verdict_rank.get(s.verdict, 4)
        return s.created_at

    filtered.sort(key=_sort_key, reverse=(order == "desc"))

    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "active": "review",
            "sessions": filtered,
            "counts": counts,
            "queue": queue,
            "task": task,
            "sort": sort,
            "order": order,
            "q": q or "",
        },
    )


@router.get("/review/{session_id}", response_class=HTMLResponse)
async def review_session(
    request: Request,
    session_id: str,
    queue: str = Query("needs_review"),
    task: Optional[str] = Query(None),
):
    sessions = scan_sessions(_data_dir())
    counts = summary_counts(sessions)
    s = find_session(sessions, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    q_sessions = filter_queue(sessions, queue=queue, task=task)
    prev_id, next_id = neighbor_ids(q_sessions, session_id)

    q_ids = [qs.session_id for qs in q_sessions]
    position = q_ids.index(session_id) + 1 if session_id in q_ids else None
    total_in_queue = len(q_ids)

    panel = load_episode_panel(Path(s.path) / "aligned" / "episode.parquet") if s.has_episode else None

    # Kick off frame extraction for smooth scrubbing (idempotent, background).
    scrub_cache.ensure_extracting(Path(s.path))

    return templates.TemplateResponse(
        "review_detail.html",
        {
            "request": request,
            "active": "review",
            "session": s,
            "counts": counts,
            "queue": queue,
            "task": task,
            "prev_id": prev_id,
            "next_id": next_id,
            "position": position,
            "total_in_queue": total_in_queue,
            "panel": panel,
        },
    )


@router.post("/review/{session_id}/decision")
async def review_decide(
    session_id: str,
    decision: str = Form(...),
    notes: str = Form(""),
    queue: str = Form("needs_review"),
    task: str = Form(""),
):
    """Persist a human decision and advance to the next session in the queue."""
    if decision not in ("approve", "reject", "force_include", "clear"):
        raise HTTPException(status_code=400, detail=f"Invalid decision: {decision}")

    sessions = scan_sessions(_data_dir())
    s = find_session(sessions, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    save_review(
        Path(s.path),
        decision=None if decision == "clear" else decision,  # type: ignore[arg-type]
        notes=notes,
    )

    # Rescan (queue membership may change) and advance
    sessions = scan_sessions(_data_dir())
    q_sessions = filter_queue(sessions, queue=queue, task=task or None)
    ids = [qs.session_id for qs in q_sessions]
    if session_id in ids:
        i = ids.index(session_id)
        target = ids[i + 1] if i + 1 < len(ids) else (ids[0] if ids else None)
    else:
        target = ids[0] if ids else None

    qs_str = f"?queue={queue}"
    if task:
        qs_str += f"&task={task}"
    if target is None:
        return RedirectResponse(url=f"/review{qs_str}", status_code=303)
    return RedirectResponse(url=f"/review/{target}{qs_str}", status_code=303)


@router.get("/review/{session_id}/video")
async def review_video(session_id: str):
    sessions = scan_sessions(_data_dir())
    s = find_session(sessions, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    video_path = Path(s.path) / "raw" / "raw_video.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="raw_video.mp4 not found")
    return FileResponse(video_path, media_type="video/mp4")


def _session_dir_fast(session_id: str) -> Path:
    """Direct path lookup — avoid the full scan_sessions cost on hot paths."""
    if not (session_id.startswith("demo_") or session_id.startswith("map_")):
        raise HTTPException(status_code=400, detail="invalid session id")
    p = _data_dir() / session_id
    if not p.is_dir():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return p


@router.get("/review/{session_id}/frame/{index}")
async def review_frame(session_id: str, index: int):
    """Frame JPEG at a specific index.

    Served from the on-disk scrub cache if present (populated by ffmpeg in the
    background on detail-page load); falls back to a PyAV decode while the
    cache is still warming.
    """
    session_dir = _session_dir_fast(session_id)

    cached = scrub_cache.get_cached_frame(session_dir, index)
    if cached:
        return Response(content=cached, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})

    jpeg = frame_from_mp4(session_dir / "raw" / "raw_video.mp4", index)
    if not jpeg:
        raise HTTPException(status_code=404, detail=f"frame {index} not found")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@router.get("/review/{session_id}/scrub_status")
async def review_scrub_status(session_id: str):
    """Polled by the UI to show cache-warming progress."""
    session_dir = _session_dir_fast(session_id)
    st = scrub_cache.status(session_dir)
    import yaml
    meta_path = session_dir / "meta.yaml"
    n_total = 0
    if meta_path.exists():
        try:
            meta = yaml.safe_load(meta_path.read_text()) or {}
            n_total = int((meta.get("quality") or {}).get("metrics", {}).get("n_frames", 0))
        except Exception:
            pass
    st["n_total"] = n_total
    return st
