"""Append-only audit trail of scored requests, in SQLite.

One row per scored member: request id, position in the batch, UTC timestamp, model
version, SHA-256 of the canonical input, calibrated probability and tier. The inputs
themselves are never stored. The hash is enough to answer "was this exact input
scored, when, and by which model" without keeping member data in a second place.

The file path comes from RISKMODEL_AUDIT_DB (default audit/audit.db under the repo).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from riskmodel import config

ENV = "RISKMODEL_AUDIT_DB"

SCHEMA = """
create table if not exists score_audit (
    request_id    text    not null,
    item_index    integer not null,
    scored_at_utc text    not null,
    model_version text    not null,
    input_sha256  text    not null,
    probability   real    not null,
    tier          text    not null,
    primary key (request_id, item_index)
)
"""


def default_path() -> Path:
    value = os.environ.get(ENV)
    return Path(value) if value else config.ROOT / "audit" / "audit.db"


def canonical_input(values: list[float | None]) -> str:
    """The input as text, the same for any two requests that mean the same thing.

    Values are in model feature order, so key order and omitted keys in the request do
    not matter. Numbers are written as floats, so 46 and 46.0 hash the same.
    """
    return json.dumps([None if v is None else float(v) for v in values], separators=(",", ":"))


def input_sha256(values: list[float | None]) -> str:
    return hashlib.sha256(canonical_input(values).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AuditLog:
    """One connection, guarded by a lock. Each request is written in one transaction."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.execute("pragma journal_mode=wal")
        self._con.execute(SCHEMA)
        self._con.commit()

    def append(self, request_id: str, model_version: str, rows: list[tuple[str, float, str]]):
        """rows: (input_sha256, probability, tier) for each member, in request order."""
        now = utc_now()
        records = [
            (request_id, i, now, model_version, digest, float(p), tier)
            for i, (digest, p, tier) in enumerate(rows)
        ]
        with self._lock, self._con:
            self._con.executemany("insert into score_audit values (?, ?, ?, ?, ?, ?, ?)", records)

    def count(self) -> int:
        with self._lock:
            return int(self._con.execute("select count(*) from score_audit").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._con.close()
