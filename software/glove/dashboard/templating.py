"""Shared Jinja2 templates with cache-busting `asset()` helper.

Cache-busting is url-suffix style: `asset('review.js')` → `/static/review.js?v=<mtime>`.
Prevents stale-cache hits in dev while letting browsers still disk-cache across
unchanged asset versions.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

DASHBOARD_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"


def _asset(name: str) -> str:
    p = STATIC_DIR / name
    try:
        return f"/static/{name}?v={int(p.stat().st_mtime)}"
    except OSError:
        return f"/static/{name}"


def make_templates() -> Jinja2Templates:
    t = Jinja2Templates(directory=str(TEMPLATE_DIR))
    t.env.globals["asset"] = _asset
    return t
