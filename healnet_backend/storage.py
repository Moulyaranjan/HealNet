"""
SQLite storage for HEALNET:
- Sensor readings (with deterministic anomaly & ML predictions)
- Node status history
- Fault lifecycle tracking
- Anomaly history
- System events & recovery events
- Edge-computing system metrics (inter-process sharing between processor and API)
"""
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Optional

log = logging.getLogger("healnet.storage")

CURRENT_SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER PRIMARY KEY,
    applied_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL,
    ts REAL NOT NULL,                 -- UTC timestamp when bridge received packet
    seq INTEGER, boot INTEGER, uptime_ms INTEGER,
    temp REAL, hum REAL, light INTEGER,
    valid INTEGER, dht_ok INTEGER, ldr_ok INTEGER,
    rssi INTEGER, via INTEGER, hops INTEGER,
    flags TEXT,                       -- comma list: gap,delayed,out_of_order,reboot,...
    anomaly INTEGER DEFAULT 0, anomaly_score REAL, anomaly_class TEXT,
    ml_anomaly INTEGER DEFAULT 0, ml_score REAL
);
CREATE INDEX IF NOT EXISTS idx_readings_node_ts ON readings(node_id, ts);
CREATE INDEX IF NOT EXISTS idx_readings_node_seq ON readings(node_id, seq);

CREATE TABLE IF NOT EXISTS status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER NOT NULL,
    old_status TEXT, new_status TEXT, reason TEXT,
    seq INTEGER
);
CREATE INDEX IF NOT EXISTS idx_status_events_node_ts ON status_events(node_id, ts);

CREATE TABLE IF NOT EXISTS faults (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER,
    fault_code INTEGER NOT NULL, fault_name TEXT, severity TEXT,
    detail TEXT, sensor_type TEXT,
    active INTEGER DEFAULT 1, cleared_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_faults_active ON faults(active, node_id, fault_code);
CREATE INDEX IF NOT EXISTS idx_faults_ts ON faults(ts DESC);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER NOT NULL,
    feature TEXT, method TEXT, score REAL, classification TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_anomalies_node_ts ON anomalies(node_id, ts);

CREATE TABLE IF NOT EXISTS route_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER NOT NULL,
    via INTEGER, hops INTEGER, state TEXT
);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event_type TEXT NOT NULL,   -- mqtt_connect, mqtt_disconnect, bridge_online, backend_startup, etc.
    detail TEXT,
    severity TEXT DEFAULT 'INFO'
);
CREATE INDEX IF NOT EXISTS idx_sys_events_ts ON system_events(ts DESC);

CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER,
    method TEXT, trigger_type TEXT, attempt INTEGER, result TEXT, duration_s REAL, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_recovery_events_ts ON recovery_events(ts DESC);

CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    ts REAL NOT NULL,
    metrics_json TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._run_migrations()

    def close(self):
        with self._lock:
            try:
                self._db.commit()
                self._db.close()
            except Exception:
                pass

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _run_migrations(self):
        """Safe additive migrations to preserve existing databases."""
        with self._lock:
            # Check readings table columns
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(readings)").fetchall()}
            if "ml_anomaly" not in columns:
                self._db.execute("ALTER TABLE readings ADD COLUMN ml_anomaly INTEGER DEFAULT 0")
            if "ml_score" not in columns:
                self._db.execute("ALTER TABLE readings ADD COLUMN ml_score REAL")

            # Check faults table columns
            f_cols = {row[1] for row in self._db.execute("PRAGMA table_info(faults)").fetchall()}
            if "sensor_type" not in f_cols:
                self._db.execute("ALTER TABLE faults ADD COLUMN sensor_type TEXT")

            # Check status_events table columns
            s_cols = {row[1] for row in self._db.execute("PRAGMA table_info(status_events)").fetchall()}
            if "seq" not in s_cols:
                self._db.execute("ALTER TABLE status_events ADD COLUMN seq INTEGER")

            # Record schema version
            self._db.execute(
                "INSERT OR REPLACE INTO schema_info (version, applied_ts) VALUES (?, ?)",
                (CURRENT_SCHEMA_VERSION, time.time()),
            )
            self._db.commit()

    # ---- writes ----------------------------------------------------------
    def insert_reading(self, r: dict):
        self._exec(
            """INSERT INTO readings(node_id,ts,seq,boot,uptime_ms,temp,hum,light,valid,dht_ok,ldr_ok,
                                    rssi,via,hops,flags,anomaly,anomaly_score,anomaly_class,
                                    ml_anomaly,ml_score)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r.get("node") or r.get("node_id"),
                r.get("rx_ts") or r.get("ts") or time.time(),
                r.get("seq"),
                r.get("boot"),
                r.get("uptime_ms"),
                r.get("temp") if r.get("temp") is not None else r.get("temperature"),
                r.get("hum") if r.get("hum") is not None else r.get("humidity"),
                r.get("light"),
                int(bool(r.get("valid", True))),
                int(bool(r.get("dht_ok", True))),
                int(bool(r.get("ldr_ok", True))),
                r.get("rssi"),
                r.get("via"),
                r.get("hops", 0),
                ",".join(r.get("flags", [])),
                int(bool(r.get("anomaly", False))),
                r.get("anomaly_score"),
                r.get("anomaly_class"),
                int(bool(r.get("ml_anomaly", False))),
                r.get("ml_score"),
            ),
        )

    def insert_status(self, ts: float, node: int, old: str, new: str, reason: str, seq: Optional[int] = None):
        self._exec(
            "INSERT INTO status_events(ts,node_id,old_status,new_status,reason,seq) VALUES(?,?,?,?,?,?)",
            (ts, node, old, new, reason, seq),
        )

    def open_fault(self, ts: float, node: Optional[int], code: int, name: str, severity: str, detail: str, sensor_type: Optional[str] = None) -> int:
        with self._lock:
            # Check for existing active fault for this node and fault code to prevent duplicates
            if node is not None:
                existing = self._db.execute(
                    "SELECT id FROM faults WHERE active=1 AND node_id=? AND fault_code=?",
                    (node, code)
                ).fetchone()
            else:
                existing = self._db.execute(
                    "SELECT id FROM faults WHERE active=1 AND node_id IS NULL AND fault_code=?",
                    (code,)
                ).fetchone()
            if existing:
                self._db.execute(
                    "UPDATE faults SET detail=?, severity=?, ts=? WHERE id=?",
                    (detail, severity, ts, existing[0])
                )
                self._db.commit()
                return existing[0]

            cur = self._db.execute(
                """INSERT INTO faults(ts,node_id,fault_code,fault_name,severity,detail,sensor_type,active)
                   VALUES(?,?,?,?,?,?,?,1)""",
                (ts, node, code, name, severity, detail, sensor_type),
            )
            self._db.commit()
            return cur.lastrowid

    def clear_fault(self, fault_id: int, ts: float):
        self._exec("UPDATE faults SET active=0, cleared_ts=? WHERE id=?", (ts, fault_id))

    def resolve_fault(self, node: Optional[int], code: int, ts: float) -> bool:
        """Resolve any active fault matching node and code."""
        with self._lock:
            if node is not None:
                cur = self._db.execute(
                    "UPDATE faults SET active=0, cleared_ts=? WHERE active=1 AND node_id=? AND fault_code=?",
                    (ts, node, code)
                )
            else:
                cur = self._db.execute(
                    "UPDATE faults SET active=0, cleared_ts=? WHERE active=1 AND node_id IS NULL AND fault_code=?",
                    (ts, code)
                )
            self._db.commit()
            return cur.rowcount > 0

    def insert_anomaly(self, ts: float, node: int, feature: str, method: str, score: float, classification: str, detail: str):
        self._exec(
            "INSERT INTO anomalies(ts,node_id,feature,method,score,classification,detail) VALUES(?,?,?,?,?,?,?)",
            (ts, node, feature, method, score, classification, detail),
        )

    def insert_route(self, ts: float, node: int, via: Optional[int], hops: int, state: str):
        self._exec(
            "INSERT INTO route_events(ts,node_id,via,hops,state) VALUES(?,?,?,?,?)",
            (ts, node, via, hops, state),
        )

    def insert_system_event(self, event_type: str, detail: str = "", severity: str = "INFO"):
        self._exec(
            "INSERT INTO system_events(ts,event_type,detail,severity) VALUES(?,?,?,?)",
            (time.time(), event_type, detail, severity),
        )

    def insert_recovery_event(
        self,
        ts: float,
        node_id: Optional[int],
        method: str = "manual_button",
        trigger_type: str = "coordinator_button",
        attempt: int = 1,
        result: str = "SUCCESS",
        duration_s: float = 0.0,
        detail: str = "",
    ):
        self._exec(
            """INSERT INTO recovery_events(ts,node_id,method,trigger_type,attempt,result,duration_s,detail)
               VALUES(?,?,?,?,?,?,?,?)""",
            (ts, node_id, method, trigger_type, attempt, result, duration_s, detail),
        )

    def save_system_metrics(self, metrics_dict: dict):
        """Store edge metrics snapshot for inter-process reading by the API."""
        now = time.time()
        payload = json.dumps(metrics_dict, separators=(",", ":"))
        self._exec(
            "INSERT OR REPLACE INTO system_metrics (id, ts, metrics_json) VALUES (1, ?, ?)",
            (now, payload),
        )

    # ---- reads -----------------------------------------------------------
    def load_system_metrics(self) -> Optional[dict]:
        """Read latest processor metrics from SQLite (used by API process)."""
        with self._lock:
            row = self._db.execute("SELECT ts, metrics_json FROM system_metrics WHERE id=1").fetchone()
            if row:
                try:
                    data = json.loads(row["metrics_json"])
                    data["db_updated_ts"] = row["ts"]
                    return data
                except Exception:
                    return None
            return None

    def recent_readings(self, node: int, limit: int = 300) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM readings WHERE node_id=? ORDER BY ts DESC, id DESC LIMIT ?",
                (node, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def readings_range(self, node: int, start_ts: float, end_ts: float, limit: int = 1000) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM readings WHERE node_id=? AND ts>=? AND ts<=? ORDER BY ts ASC LIMIT ?",
                (node, start_ts, end_ts, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_reading(self, node: int) -> Optional[dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM readings WHERE node_id=? ORDER BY ts DESC, id DESC LIMIT 1",
                (node,),
            ).fetchone()
        return dict(row) if row else None

    def all_nodes_latest(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """SELECT r.* FROM readings r
                   INNER JOIN (
                       SELECT node_id, MAX(ts) as max_ts FROM readings GROUP BY node_id
                   ) latest ON r.node_id=latest.node_id AND r.ts=latest.max_ts
                   ORDER BY r.node_id"""
            ).fetchall()
        return [dict(r) for r in rows]

    def all_readings(self, limit: int = 100, node_id: Optional[int] = None) -> list[dict]:
        with self._lock:
            if node_id is not None:
                rows = self._db.execute(
                    "SELECT * FROM readings WHERE node_id=? ORDER BY ts DESC LIMIT ?",
                    (node_id, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM readings ORDER BY ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def active_faults(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM faults WHERE active=1 ORDER BY ts DESC").fetchall()
        return [dict(r) for r in rows]

    def fault_history(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM faults ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def node_faults(self, node_id: int, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM faults WHERE node_id=? ORDER BY ts DESC LIMIT ?", (node_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def anomaly_history(self, node: Optional[int] = None, limit: int = 100) -> list[dict]:
        with self._lock:
            if node is not None:
                rows = self._db.execute(
                    "SELECT * FROM anomalies WHERE node_id=? ORDER BY ts DESC LIMIT ?",
                    (node, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM anomalies ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    def system_events(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM system_events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def recovery_events(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM recovery_events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_node_status(self, node_id: int) -> Optional[dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM status_events WHERE node_id=? ORDER BY ts DESC LIMIT 1",
                (node_id,),
            ).fetchone()
        return dict(row) if row else None

    def close_stale_faults(self, ts: float):
        """On backend start, faults left active by a previous run are re-evaluated from scratch."""
        self._exec("UPDATE faults SET active=0, cleared_ts=? WHERE active=1", (ts,))

    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {}
            for table in (
                "readings",
                "faults",
                "anomalies",
                "status_events",
                "route_events",
                "system_events",
                "recovery_events",
            ):
                try:
                    out[table] = self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except Exception:
                    out[table] = 0
        return out

    def purge_older_than(self, days: int):
        cutoff = time.time() - days * 86400
        with self._lock:
            for table in ("readings", "status_events", "anomalies", "route_events", "system_events", "recovery_events"):
                self._db.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
            self._db.execute("DELETE FROM faults WHERE active=0 AND ts < ?", (cutoff,))
            self._db.commit()
