import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS plants (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, config TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, plant_id TEXT NOT NULL, plant_name TEXT NOT NULL, plant TEXT NOT NULL,
    status TEXT NOT NULL,
    training TEXT NOT NULL, timesteps_done INTEGER NOT NULL DEFAULT 0, error TEXT, model_id TEXT,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, plant_name TEXT NOT NULL, job_id TEXT, path TEXT NOT NULL,
    horizon INTEGER NOT NULL, evaluation TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY, request TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL
);
"""

JSON_COLUMNS = {"config", "plant", "training", "evaluation", "request", "result"}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _decode(row) -> dict | None:
        if row is None:
            return None
        return {
            k: json.loads(row[k]) if k in JSON_COLUMNS and row[k] is not None else row[k] for k in row.keys()
        }

    def insert(self, table: str, **values) -> dict:
        values = {"id": new_id(), "created_at": now()} | values
        encoded = {k: json.dumps(v) if k in JSON_COLUMNS else v for k, v in values.items()}
        with self._connect() as db:
            db.execute(
                f"INSERT INTO {table} ({', '.join(encoded)}) VALUES ({', '.join('?' * len(encoded))})",
                list(encoded.values()),
            )
        return self.get(table, values["id"])

    def update(self, table: str, id: str, **values) -> dict | None:
        encoded = {k: json.dumps(v) if k in JSON_COLUMNS else v for k, v in values.items()}
        with self._connect() as db:
            db.execute(
                f"UPDATE {table} SET {', '.join(f'{k} = ?' for k in encoded)} WHERE id = ?",
                [*encoded.values(), id],
            )
        return self.get(table, id)

    def get(self, table: str, id: str) -> dict | None:
        with self._connect() as db:
            return self._decode(db.execute(f"SELECT * FROM {table} WHERE id = ?", [id]).fetchone())

    def list(self, table: str, **where) -> list[dict]:
        clause = " AND ".join(f"{k} = ?" for k in where)
        sql = f"SELECT * FROM {table}" + (f" WHERE {clause}" if clause else "") + " ORDER BY created_at DESC"
        with self._connect() as db:
            return [self._decode(r) for r in db.execute(sql, list(where.values())).fetchall()]

    def delete(self, table: str, id: str) -> bool:
        with self._connect() as db:
            return db.execute(f"DELETE FROM {table} WHERE id = ?", [id]).rowcount > 0

    def get_setting(self, key: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key = ?", [key]).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str):
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", [key, value])
