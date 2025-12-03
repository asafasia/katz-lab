import sqlite3

DB_PATH = "calibration.db"

schema = """
CREATE TABLE IF NOT EXISTS qubits (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    qpu_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibrations (
    id INTEGER PRIMARY KEY,
    qubit_id INTEGER NOT NULL,
    kind TEXT NOT NULL,              -- e.g. 'readout', 'drive_freq', 'drag'
    params_json TEXT NOT NULL,       -- JSON string with arbitrary parameters
    created_at TEXT NOT NULL,        -- ISO timestamp
    created_by TEXT,                 -- your username
    git_hash TEXT,                   -- optional for traceability
    FOREIGN KEY (qubit_id) REFERENCES qubits(id)
);
"""

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    print(f"Initialized DB at {DB_PATH}")

if __name__ == "__main__":
    main()