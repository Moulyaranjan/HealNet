"""SQLite storage for readings, node status history, faults, anomalies and routes."""
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL,
    ts REAL NOT NULL,                 -- laptop time when the bridge received the packet
    seq INTEGER, boot INTEGER, uptime_ms INTEGER,
    temp REAL, hum REAL, light INTEGER,
    valid INTEGER, dht_ok INTEGER, ldr_ok INTEGER,
    rssi INTEGER, via INTEGER, hops INTEGER,
    flags TEXT,                       -- comma list: gap,delayed,out_of_order,reboot,...
    anomaly INTEGER DEFAULT 0, anomaly_score REAL, anomaly_class TEXT
);
CREATE INDEX IF NOT EXISTS idx_readings_node_ts ON readings(node_id, ts);

CREATE TABLE IF NOT EXISTS status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER NOT NULL,
    old_status TEXT, new_status TEXT, reason TEXT
);

CREATE TABLE IF NOT EXISTS faults (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER,
    fault_code INTEGER NOT NULL, fault_name TEXT, severity TEXT,
    detail TEXT, active INTEGER DEFAULT 1, cleared_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_faults_active ON faults(active, node_id, fault_code);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER NOT NULL,
    feature TEXT, method TEXT, score REAL, classification TEXT, detail TEXT
);

CREATE TABLE IF NOT EXISTS route_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER NOT NULL,
    via INTEGER, hops INTEGER, state TEXT
);

-- RESERVED: filled by the recovery subsystem in a later phase.
CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, node_id INTEGER,
    method TEXT, trigger_type TEXT, attempt INTEGER, result TEXT, duration_s REAL, detail TEXT
);
"""


class Storage:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self):
        with self._lock:
            self._db.commit()
            self._db.close()

    def _exec(self, sql, args=()):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    # ---- writes ----------------------------------------------------------
    def insert_reading(self, r):
        self._exec(
            """INSERT INTO readings(node_id,ts,seq,boot,uptime_ms,temp,hum,light,valid,dht_ok,ldr_ok,
                                    rssi,via,hops,flags,anomaly,anomaly_score,anomaly_class)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r["node"], r["rx_ts"], r.get("seq"), r.get("boot"), r.get("uptime_ms"),
             r.get("temp"), r.get("hum"), r.get("light"),
             int(bool(r.get("valid", True))), int(bool(r.get("dht_ok", True))), int(bool(r.get("ldr_ok", True))),
             r.get("rssi"), r.get("via"), r.get("hops", 0),
             ",".join(r.get("flags", [])), int(bool(r.get("anomaly", False))),
             r.get("anomaly_score"), r.get("anomaly_class")))

    def insert_status(self, ts, node, old, new, reason):
        self._exec("INSERT INTO status_events(ts,node_id,old_status,new_status,reason) VALUES(?,?,?,?,?)",
                   (ts, node, old, new, reason))

    def open_fault(self, ts, node, code, name, severity, detail):
        cur = self._exec(
            "INSERT INTO faults(ts,node_id,fault_code,fault_name,severity,detail) VALUES(?,?,?,?,?,?)",
            (ts, node, code, name, severity, detail))
        return cur.lastrowid

    def clear_fault(self, fault_id, ts):
        self._exec("UPDATE faults SET active=0, cleared_ts=? WHERE id=?", (ts, fault_id))

    def insert_anomaly(self, ts, node, feature, method, score, classification, detail):
        self._exec(
            "INSERT INTO anomalies(ts,node_id,feature,method,score,classification,detail) VALUES(?,?,?,?,?,?,?)",
            (ts, node, feature, method, score, classification, detail))

    def insert_route(self, ts, node, via, hops, state):
        self._exec("INSERT INTO route_events(ts,node_id,via,hops,state) VALUES(?,?,?,?,?)",
                   (ts, node, via, hops, state))

    # ---- reads -----------------------------------------------------------
    def recent_readings(self, node, limit=300):
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM readings WHERE node_id=? ORDER BY ts DESC, id DESC LIMIT ?",
                (node, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def active_faults(self):
        with self._lock:
            rows = self._db.execute("SELECT * FROM faults WHERE active=1").fetchall()
        return [dict(r) for r in rows]

    def close_stale_faults(self, ts):
        """On backend start, faults left active by a previous run are re-evaluated from scratch."""
        self._exec("UPDATE faults SET active=0, cleared_ts=? WHERE active=1", (ts,))

    def counts(self):
        with self._lock:
            out = {}
            for table in ("readings", "faults", "anomalies", "status_events", "route_events"):
                out[table] = self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return out

    def purge_older_than(self, days):
        cutoff = time.time() - days * 86400
        with self._lock:
            for table in ("readings", "status_events", "anomalies", "route_events"):
                self._db.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
            self._db.execute("DELETE FROM faults WHERE active=0 AND ts < ?", (cutoff,))
            self._db.commit()
