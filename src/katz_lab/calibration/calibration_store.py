# calibration/calibration_store.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path

class CalibrationStore:
    def __init__(self, db_path="calibration/calibration.db"):
        Path(db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS calibrations (
            id INTEGER PRIMARY KEY,
            qubit TEXT NOT NULL,
            kind TEXT NOT NULL,
            params_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        self.conn.commit()

    def insert(self, qubit, kind, params):
        self.conn.execute(
            "INSERT INTO calibrations (qubit, kind, params_json, created_at) VALUES (?, ?, ?, ?)",
            (qubit, kind, json.dumps(params), datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def latest(self, qubit, kind):
        row = self.conn.execute(
            "SELECT params_json FROM calibrations WHERE qubit=? AND kind=? ORDER BY created_at DESC LIMIT 1",
            (qubit, kind)
        ).fetchone()
        return json.loads(row["params_json"]) if row else None