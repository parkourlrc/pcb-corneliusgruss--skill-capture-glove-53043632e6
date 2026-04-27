"""FastAPI dashboard app and CLI entry point.

Browse-only review UI. Walks a session directory, presents per-session
quality metrics + decision controls, and lets you scrub through video, trajectory,
and proprio panels for any captured demo.

No live hardware: the capture and recording paths from the original dashboard
are excluded; this is a post-hoc audit tool only.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from glove.dashboard.config import DATA_DIR, DB_NAME, WEB_HOST, WEB_PORT

logger = logging.getLogger(__name__)

# Module-level state, configured by create_app()
session_db = None
data_dir = DATA_DIR

DASHBOARD_DIR = Path(__file__).parent


def create_app(data: Optional[Path] = None) -> FastAPI:
    """Create and configure the FastAPI app."""
    global session_db, data_dir

    if data:
        data_dir = data
    data_dir.mkdir(parents=True, exist_ok=True)

    from glove.dashboard.db import SessionDB

    db_path = data_dir.parent / DB_NAME
    session_db = SessionDB(db_path)

    app = FastAPI(title="Skill-Capture Glove Dashboard")

    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static")

    from glove.dashboard.routes import review
    app.include_router(review.router)

    @app.on_event("startup")
    async def startup():
        if session_db:
            synced = session_db.sync_from_filesystem(data_dir)
            logger.info("Session DB synced: %d sessions updated", synced)

    @app.on_event("shutdown")
    async def shutdown():
        if session_db:
            session_db.close()

    from glove.dashboard.templating import make_templates
    templates = make_templates()

    _months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def fmt_session_date(name: str) -> str:
        """Render session_2026-03-01_201023 → 'Mar 1, 20:10'."""
        try:
            parts = name.split('_')
            year, month, day = parts[1].split('-')
            time = parts[2] if len(parts) > 2 else '000000'
            return f"{_months[int(month) - 1]} {int(day)}, {time[:2]}:{time[2:4]}"
        except Exception:
            return name

    templates.env.filters["session_date"] = fmt_session_date

    def fmt_session_title(session) -> str:
        """Return a human-friendly session title when one exists."""
        if isinstance(session, dict):
            display_name = str(session.get("display_name", "")).strip()
            if display_name:
                return display_name
            name = str(session.get("name", ""))
            return fmt_session_date(name)
        if isinstance(session, str):
            return fmt_session_date(session)
        return str(session)

    templates.env.filters["session_title"] = fmt_session_title

    @app.get("/")
    async def index():
        return RedirectResponse(url="/review", status_code=302)

    return app


def main():
    parser = argparse.ArgumentParser(description="Skill-Capture Glove Dashboard")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"Session data directory (default: {DATA_DIR})")
    parser.add_argument("--port", type=int, default=WEB_PORT,
                        help=f"Web server port (default: {WEB_PORT})")
    parser.add_argument("--host", default=WEB_HOST,
                        help=f"Web server host (default: {WEB_HOST})")
    args = parser.parse_args()

    app = create_app(data=args.data_dir)

    print(f"\nSkill-Capture Glove Dashboard")
    print(f"  URL:  http://localhost:{args.port}")
    print(f"  Data: {args.data_dir}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
