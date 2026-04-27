"""SQLite session metadata cache.

The filesystem (session directories + metadata.json) remains the source of truth.
This database is a cache for fast listing, filtering, and aggregation. If it gets
corrupted or deleted, restart the dashboard — it will be rebuilt from disk.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    name TEXT PRIMARY KEY,
    task TEXT NOT NULL DEFAULT '',
    display_name TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    hardware_version TEXT NOT NULL DEFAULT '',
    num_frames INTEGER NOT NULL DEFAULT 0,
    num_poses INTEGER NOT NULL DEFAULT 0,
    num_encoders INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL,
    slam_processed INTEGER,
    has_imu INTEGER NOT NULL DEFAULT 0,
    imu_stale INTEGER,
    review_status TEXT NOT NULL DEFAULT 'pending',
    quality_score REAL,
    quality_json TEXT,
    thumbnail_path TEXT,
    metadata_json TEXT,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_task ON sessions(task);
CREATE INDEX IF NOT EXISTS idx_sessions_review ON sessions(review_status);
CREATE INDEX IF NOT EXISTS idx_sessions_quality ON sessions(quality_score);
"""


class SessionDB:
    """SQLite cache for session metadata.

    Thread-safe via a write lock. Reads are lock-free (SQLite WAL mode).

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path), check_same_thread=False, isolation_level="DEFERRED"
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """Apply lightweight schema migrations for older SQLite files."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "display_name" not in cols:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN display_name TEXT NOT NULL DEFAULT ''"
            )
        if "tags" not in cols:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN tags TEXT NOT NULL DEFAULT ''"
            )
        if "hardware_version" not in cols:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN hardware_version TEXT NOT NULL DEFAULT ''"
            )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def upsert_session(self, name: str, **fields: object) -> None:
        """Insert or update a session row.

        Parameters
        ----------
        name : str
            Session directory name (e.g. 'session_2026-03-30_191214').
        **fields
            Column values to set. Unknown columns are silently ignored.
        """
        fields["updated_at"] = time.time()
        valid_cols = {
            "task", "display_name", "tags", "hardware_version",
            "num_frames", "num_poses", "num_encoders",
            "duration_seconds", "slam_processed", "has_imu", "imu_stale",
            "review_status", "quality_score", "quality_json",
            "thumbnail_path", "metadata_json", "updated_at",
        }
        filtered = {k: v for k, v in fields.items() if k in valid_cols}

        cols = ["name"] + list(filtered.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in filtered.keys())

        sql = (
            f"INSERT INTO sessions ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(name) DO UPDATE SET {updates}"
        )
        values = [name] + list(filtered.values())

        with self._lock:
            self._conn.execute(sql, values)
            self._conn.commit()

    def get_session(self, name: str) -> Optional[dict]:
        """Get a single session row by name.

        Returns
        -------
        dict or None
            Session data as a dict, or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self,
        task: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        """List sessions with optional filters, ordered newest first.

        Parameters
        ----------
        task : str, optional
            Filter by task name.
        status : str, optional
            Filter by review status ('approved', 'rejected', 'pending').

        Returns
        -------
        list[dict]
            Session rows as dicts.
        """
        clauses = []
        params: list[object] = []
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        if status is not None:
            clauses.append("review_status = ?")
            params.append(status)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM sessions{where} ORDER BY name DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def update_review_status(self, name: str, status: str) -> None:
        """Update the review status for a session.

        Parameters
        ----------
        name : str
            Session name.
        status : str
            New status ('approved', 'rejected', 'pending').
        """
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET review_status = ?, updated_at = ? WHERE name = ?",
                (status, time.time(), name),
            )
            self._conn.commit()

    def update_quality(self, name: str, quality: dict) -> None:
        """Update quality score and JSON for a session.

        Parameters
        ----------
        name : str
            Session name.
        quality : dict
            Quality dict (must contain 'score' key).
        """
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET quality_score = ?, quality_json = ?, updated_at = ? WHERE name = ?",
                (quality.get("score"), json.dumps(quality), time.time(), name),
            )
            self._conn.commit()

    def delete_session(self, name: str) -> None:
        """Remove a session from the database.

        Parameters
        ----------
        name : str
            Session name to delete.
        """
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE name = ?", (name,))
            self._conn.commit()

    def sync_from_filesystem(self, data_dir: Path) -> int:
        """Sync database with session directories on disk.

        Compares metadata.json mtime against the DB updated_at timestamp.
        Only re-reads sessions whose metadata has changed. Removes DB rows
        for sessions that no longer exist on disk.

        Parameters
        ----------
        data_dir : Path
            Directory containing session_* subdirectories.

        Returns
        -------
        int
            Number of sessions synced (inserted or updated).
        """
        if not data_dir.exists():
            return 0

        on_disk: set[str] = set()
        synced = 0

        for d in sorted(data_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("session_"):
                continue
            on_disk.add(d.name)

            # Support new format (meta/episode.json) and legacy (metadata.json)
            episode_meta_path = d / "meta" / "episode.json"
            legacy_meta_path = d / "metadata.json"

            if episode_meta_path.exists():
                meta_path = episode_meta_path
            elif legacy_meta_path.exists():
                meta_path = legacy_meta_path
            else:
                continue

            disk_mtime = meta_path.stat().st_mtime
            db_row = self.get_session(d.name)
            if db_row and db_row["updated_at"] >= disk_mtime:
                continue

            try:
                meta = json.loads(meta_path.read_text())

                # New format: read from info.json + episode.json
                info_path = d / "meta" / "info.json"
                if info_path.exists():
                    info = json.loads(info_path.read_text())
                    num_frames = info.get("num_frames", 0)
                    duration_seconds = info.get("duration_s")
                    has_imu = (d / "data" / "imu.parquet").exists() or (d / "data" / "episode.parquet").exists()
                else:
                    # No meta/info.json: derive from native Parquet if present
                    info = {}
                    episode_parquet = d / "data" / "episode.parquet"
                    if episode_parquet.exists():
                        import pyarrow.parquet as pq

                        num_frames = pq.read_metadata(episode_parquet).num_rows
                    else:
                        num_frames = 0
                    duration_seconds = meta.get("durationSeconds")
                    has_imu = (d / "data" / "imu.parquet").exists() or episode_parquet.exists()

                # Count poses (new or legacy format)
                num_poses = 0
                poses_parquet = d / "data" / "poses.parquet"
                poses_json = d / "poses.json"
                if poses_parquet.exists():
                    import pyarrow.parquet as pq
                    num_poses = pq.read_metadata(poses_parquet).num_rows
                elif poses_json.exists():
                    try:
                        poses = json.loads(poses_json.read_text())
                        num_poses = len(poses)
                    except Exception:
                        pass

                # Count encoders
                num_encoders = num_frames if (d / "data" / "episode.parquet").exists() else 0
                if num_encoders == 0:
                    enc_path = d / "encoders.json"
                    if enc_path.exists():
                        try:
                            encoders = json.loads(enc_path.read_text())
                            num_encoders = len(encoders)
                        except Exception:
                            pass

                quality = meta.get("quality")

                # Thumbnail — generate if missing
                from glove.dashboard.thumbnails import ensure_thumbnail
                thumb_result = ensure_thumbnail(d)
                thumbnail = "thumbnail.jpg" if thumb_result else None

                # Read hardware version from info.json
                hw_version = ""
                hw_info = info.get("hardware", {})
                if hw_info.get("daq_board") == "stm32":
                    hw_version = "v3"
                elif hw_info.get("daq_board") == "esp32":
                    hw_version = "v2"
                elif hw_info.get("glove_version"):
                    hw_version = hw_info["glove_version"]

                # Read tags from episode.json
                tags_list = meta.get("tags", [])
                tags_str = ",".join(tags_list) if isinstance(tags_list, list) else ""

                self.upsert_session(
                    name=d.name,
                    task=meta.get("task", ""),
                    display_name=meta.get("display_name", ""),
                    tags=tags_str,
                    hardware_version=hw_version,
                    num_frames=num_frames,
                    num_poses=num_poses,
                    num_encoders=num_encoders,
                    duration_seconds=duration_seconds,
                    slam_processed=1 if meta.get("slamProcessed") or meta.get("has_poses") else 0,
                    has_imu=1 if has_imu else 0,
                    imu_stale=1 if (quality and quality.get("imu_stale")) else None,
                    review_status=meta.get("review_status", "pending"),
                    quality_score=quality.get("score") if quality else None,
                    quality_json=json.dumps(quality) if quality else None,
                    thumbnail_path=thumbnail,
                    metadata_json=json.dumps(meta),
                )
                synced += 1
            except Exception as e:
                logger.warning("Failed to sync session %s: %s", d.name, e)

        # Remove orphaned DB rows
        db_names = {r["name"] for r in self.list_sessions()}
        orphans = db_names - on_disk
        for name in orphans:
            self.delete_session(name)
            logger.info("Removed orphaned DB entry: %s", name)

        return synced
