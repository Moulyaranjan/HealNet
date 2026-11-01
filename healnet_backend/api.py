"""
HEALNET Local REST API (FastAPI + Uvicorn).

Provides read-only dashboard-facing endpoints backed by the SQLite database.
All writes go through the normal MQTT → processor → storage pipeline; the API
only reads.  Real-time push is via MQTT; polling endpoints are available here.

Run:
    python api.py                           # default: http://127.0.0.1:8000
    python api.py --host 0.0.0.0 --port 8080

Environment overrides:
    HEALNET_DB_PATH   – path to healnet.db (falls back to config.DB_PATH)
    HEALNET_API_HOST  – bind host
    HEALNET_API_PORT  – bind port
"""
import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import config  # noqa: E402 (local module)

log = logging.getLogger("healnet.api")

# ---------------------------------------------------------------------------
# Lazy import: FastAPI is optional so the rest of the backend still works
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    _HAVE_FASTAPI = True
except ImportError:
    _HAVE_FASTAPI = False
    FastAPI = object  # type: ignore

from storage import Storage
from edge_metrics import get_metrics

# ---------------------------------------------------------------------------
# Shared DB instance (opened once at startup, closed on shutdown)
# ---------------------------------------------------------------------------
_db: Optional[Storage] = None
_start_ts = time.time()


def _get_db() -> Storage:
    global _db
    if _db is None:
        db_path = os.environ.get("HEALNET_DB_PATH", config.DB_PATH)
        log.info("API: opening database at %s (lazy-init)", db_path)
        _db = Storage(db_path)
    return _db


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    db_path = os.environ.get("HEALNET_DB_PATH", config.DB_PATH)
    log.info("API: opening database at %s", db_path)
    _db = Storage(db_path)
    yield
    if _db:
        _db.close()
    log.info("API: database closed")


app = FastAPI(
    title="HEALNET Local API",
    description=(
        "Read-only REST API for the HEALNET IoT monitoring backend. "
        "All data is served from the local SQLite database. "
        "Real-time streams are available via the MQTT broker on healnet/#."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dashboard may run on a different port; lock down in production
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _ok(data: Any) -> dict:
    return {"status": "ok", "ts": round(time.time(), 3), "data": data}


def _err(msg: str, code: int = 400):
    raise HTTPException(status_code=code, detail=msg)


# ---------------------------------------------------------------------------
# Sensor endpoints
# ---------------------------------------------------------------------------
@app.get("/api/sensors", summary="Latest reading for all nodes")
@app.get("/api/readings", summary="Latest reading for all nodes (alias)")
@app.get("/api/readings/latest", summary="Latest reading for all nodes (alias)")
def get_sensors():
    """Return the most recent reading for every known sensor node."""
    db = _get_db()
    rows = db.all_nodes_latest()
    return _ok(rows)


@app.get("/api/readings/all", summary="All historical readings across nodes")
def get_all_readings(limit: int = Query(default=100, ge=1, le=5000)):
    """Return sensor readings across all nodes ordered by timestamp descending."""
    db = _get_db()
    rows = db.all_readings(limit)
    return _ok({"count": len(rows), "readings": rows})


@app.get("/api/sensors/{node_id}/latest", summary="Latest reading for a single node")
@app.get("/api/readings/{node_id}", summary="Latest reading for a single node (alias)")
def get_sensor_latest(node_id: int):
    """Return the most recent reading for the specified node."""
    if node_id not in config.NODE_IDS:
        _err(f"Unknown node_id {node_id}. Valid: {list(config.NODE_IDS)}", 404)
    db = _get_db()
    row = db.latest_reading(node_id)
    if row is None:
        _err(f"No readings found for node {node_id}", 404)
    return _ok(row)


@app.get("/api/sensors/{node_id}/history", summary="Historical readings for a node")
@app.get("/api/nodes/{node_id}/readings", summary="Historical readings for a node (alias)")
def get_sensor_history(
    node_id: int,
    limit: int = Query(default=100, ge=1, le=10000),
    start: Optional[float] = Query(default=None, description="Start Unix timestamp"),
    end: Optional[float] = Query(default=None, description="End Unix timestamp"),
):
    """
    Return historical sensor readings for a node.

    - **limit**: maximum number of records (default 100, max 10 000)
    - **start** / **end**: optional Unix timestamp range filter
    """
    if node_id not in config.NODE_IDS:
        _err(f"Unknown node_id {node_id}. Valid: {list(config.NODE_IDS)}", 404)
    db = _get_db()
    if start is not None or end is not None:
        s = start or 0.0
        e = end or time.time()
        rows = db.readings_range(node_id, s, e, limit)
    else:
        rows = db.recent_readings(node_id, limit)
    return _ok({"node_id": node_id, "count": len(rows), "readings": rows})


# ---------------------------------------------------------------------------
# Node endpoints
# ---------------------------------------------------------------------------
def _derive_node_status(db: Storage, node_id: int, latest: Optional[dict]) -> tuple[str, Optional[float]]:
    """Derive accurate node status (ONLINE, STALE, OFFLINE, UNKNOWN) and status_since timestamp."""
    evt = db.latest_node_status(node_id)
    # Check if coordinator is reported down via active FAULT 5
    active_sys_faults = [f for f in db.active_faults() if f.get("node_id") is None and f.get("fault_code") == config.FAULT_COORDINATOR]
    if active_sys_faults:
        return "UNKNOWN", evt["ts"] if evt else None

    if latest:
        age = time.time() - latest["ts"]
        if age < config.NODE_STALE_S:
            status = "ONLINE"
        elif age < config.NODE_TIMEOUT_S:
            status = "STALE"
        else:
            status = "OFFLINE"
        status_since = evt["ts"] if evt else latest["ts"]
        return status, status_since

    if evt:
        return evt.get("new_status", "UNKNOWN"), evt["ts"]
    return "UNKNOWN", None


@app.get("/api/nodes", summary="Status overview of all nodes")
def get_nodes():
    """Return a status snapshot for every configured sensor node."""
    db = _get_db()
    out = []
    for nid in config.NODE_IDS:
        row = db.latest_reading(nid)
        status, status_since = _derive_node_status(db, nid, row)
        faults = [f for f in db.active_faults() if f.get("node_id") == nid]
        age_s = round(time.time() - row["ts"], 1) if row else None
        out.append({
            "node_id": nid,
            "status": status,
            "status_since": status_since,
            "age_s": age_s,
            "last_reading": row,
            "active_faults": faults,
            "active_fault_count": len(faults),
        })
    return _ok(out)


@app.get("/api/nodes/{node_id}/status", summary="Status for a specific node")
def get_node_status(node_id: int):
    """
    Return status, last reading and active faults for a specific node.
    Node status (ONLINE/STALE/OFFLINE/UNKNOWN) is determined by the processor
    and stored in the status_events table.
    """
    if node_id not in config.NODE_IDS:
        _err(f"Unknown node_id {node_id}. Valid: {list(config.NODE_IDS)}", 404)
    db = _get_db()
    latest = db.latest_reading(node_id)
    status, status_since = _derive_node_status(db, node_id, latest)
    faults = [f for f in db.active_faults() if f.get("node_id") == node_id]
    age_s = round(time.time() - latest["ts"], 1) if latest else None
    return _ok({
        "node_id": node_id,
        "status": status,
        "status_since": status_since,
        "last_reading": latest,
        "age_s": age_s,
        "active_faults": faults,
        "active_fault_count": len(faults),
    })


@app.get("/api/nodes/{node_id}/faults", summary="Fault history for a specific node")
def get_node_faults(
    node_id: int,
    limit: int = Query(default=50, ge=1, le=1000),
):
    """Return active and historical fault records for a specific node."""
    if node_id not in config.NODE_IDS:
        _err(f"Unknown node_id {node_id}. Valid: {list(config.NODE_IDS)}", 404)
    db = _get_db()
    rows = db.node_faults(node_id, limit)
    return _ok({"node_id": node_id, "count": len(rows), "faults": rows})


# ---------------------------------------------------------------------------
# Fault endpoints
# ---------------------------------------------------------------------------
@app.get("/api/faults", summary="All faults (active + historical)")
def get_faults(limit: int = Query(default=50, ge=1, le=1000)):
    """Return the most recent fault records, both active and cleared."""
    db = _get_db()
    rows = db.fault_history(limit)
    return _ok({"count": len(rows), "faults": rows})


@app.get("/api/faults/active", summary="Currently active faults")
def get_faults_active():
    """Return all faults that are currently active (not yet cleared)."""
    db = _get_db()
    rows = db.active_faults()
    return _ok({"count": len(rows), "faults": rows})


@app.get("/api/faults/history", summary="Historical fault records")
def get_faults_history(limit: int = Query(default=100, ge=1, le=5000)):
    """Return cleared fault records ordered by detection timestamp descending."""
    db = _get_db()
    rows = db.fault_history(limit)
    return _ok({"count": len(rows), "faults": rows})


# ---------------------------------------------------------------------------
# Anomaly endpoints
# ---------------------------------------------------------------------------
@app.get("/api/anomalies", summary="Recent anomalies (all nodes)")
def get_anomalies(limit: int = Query(default=50, ge=1, le=1000)):
    """Return the most recent anomaly records across all nodes."""
    db = _get_db()
    rows = db.anomaly_history(limit=limit)
    return _ok({"count": len(rows), "anomalies": rows})


@app.get("/api/anomalies/history", summary="Anomaly history for a specific node")
def get_anomaly_history(
    node_id: Optional[int] = Query(default=None, description="Filter by node ID"),
    limit: int = Query(default=100, ge=1, le=5000),
):
    """Return anomaly history, optionally filtered by node."""
    if node_id is not None and node_id not in config.NODE_IDS:
        _err(f"Unknown node_id {node_id}. Valid: {list(config.NODE_IDS)}", 404)
    db = _get_db()
    rows = db.anomaly_history(node=node_id, limit=limit)
    return _ok({"count": len(rows), "anomalies": rows})


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------
@app.get("/api/system/health", summary="Backend health check")
@app.get("/api/health", summary="Backend health check (alias)")
def get_system_health():
    """
    Quick health check endpoint.
    Returns 200 if the API can reach the database, 503 otherwise.
    """
    db = _get_db()
    counts = db.counts()
    metrics = get_metrics()
    snap = metrics.snapshot()
    return _ok({
        "status": "healthy",
        "uptime_s": snap["uptime_s"],
        "uptime_human": snap["uptime_human"],
        "db_counts": counts,
        "messages_total": snap["messages_total"],
        "active_faults": len(db.active_faults()),
    })


@app.get("/api/system/metrics", summary="Edge-computing resource metrics")
def get_system_metrics():
    """
    Return edge-computing metrics: CPU, memory, disk usage and processing latencies.
    Reads latest metrics saved by the background processor, enriched with local OS stats.
    """
    db = _get_db()
    proc_snap = db.load_system_metrics()
    if proc_snap:
        local_snap = get_metrics().snapshot()
        proc_snap["api_uptime_s"] = local_snap["uptime_s"]
        if "os" in local_snap and ("os" not in proc_snap or not proc_snap["os"].get("cpu_percent")):
            proc_snap["os"] = local_snap["os"]
        return _ok(proc_snap)
    snap = get_metrics().snapshot()
    return _ok(snap)


@app.get("/api/system/events", summary="System event log")
@app.get("/api/events", summary="System event log (alias)")
def get_system_events(limit: int = Query(default=50, ge=1, le=1000)):
    """Return the most recent system events (MQTT connect/disconnect, bridge status, etc.)."""
    db = _get_db()
    rows = db.system_events(limit)
    return _ok({"count": len(rows), "events": rows})


# ---------------------------------------------------------------------------
# ML endpoints
# ---------------------------------------------------------------------------
@app.get("/api/ml/status", summary="ML model status")
def get_ml_status():
    """
    Return the status of the trained ML anomaly-detection model.
    Loads metadata from disk if available.
    """
    import json
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    meta_path = os.path.join(model_dir, "anomaly_metadata.json")
    if not os.path.exists(meta_path):
        return _ok({
            "model_loaded": False,
            "note": "No trained model found. Run: python ml_pipeline.py",
        })
    with open(meta_path, "r") as fh:
        meta = json.load(fh)
    model_path = os.path.join(model_dir, "anomaly_model.joblib")
    return _ok({
        "model_loaded": os.path.exists(model_path),
        "model_path": model_path,
        "metadata": meta,
    })


@app.get("/api/ml/metrics", summary="ML evaluation metrics")
def get_ml_metrics():
    """
    Return the evaluation metrics from the last model training run.
    Reads from the metadata JSON saved alongside the model.
    """
    import json
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    meta_path = os.path.join(model_dir, "anomaly_metadata.json")
    if not os.path.exists(meta_path):
        _err("No model metadata found. Train the model first: python ml_pipeline.py", 404)
    with open(meta_path, "r") as fh:
        meta = json.load(fh)
    return _ok(meta.get("evaluation", {}))


@app.get("/api/ml/evaluation", summary="Detailed ML evaluation report")
def get_ml_evaluation():
    """
    Return the complete evaluation report comparing IsolationForest,
    LocalOutlierFactor and OneClassSVM with precision, recall, F1, accuracy, FPR, FNR.
    """
    import json
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    report_path = os.path.join(model_dir, "evaluation_report.json")
    if not os.path.exists(report_path):
        meta_path = os.path.join(model_dir, "anomaly_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
            return _ok(meta)
        _err("No evaluation report found. Train the model first: python ml_pipeline.py --train", 404)
    with open(report_path, "r") as fh:
        rep = json.load(fh)
    return _ok(rep)


@app.get("/api/system/recovery", summary="Recovery events log")
@app.get("/api/recovery", summary="Recovery events log (alias)")
def get_recovery_events(limit: int = Query(default=50, ge=1, le=1000)):
    """Return recovery events log (manual button recovery, simulated recovery, etc.)."""
    db = _get_db()
    rows = db.recovery_events(limit)
    return _ok({"count": len(rows), "events": rows})


@app.get("/api/system/summary", summary="System-wide network health summary")
def get_system_summary():
    """Return an aggregated summary of the network status, health percentage and active faults."""
    db = _get_db()
    counts = {"ONLINE": 0, "STALE": 0, "OFFLINE": 0, "UNKNOWN": 0}
    for nid in config.NODE_IDS:
        r = db.latest_reading(nid)
        st, _ = _derive_node_status(db, nid, r)
        counts[st] = counts.get(st, 0) + 1

    active_faults = db.active_faults()
    coord_faults = [f for f in active_faults if f.get("node_id") is None and f.get("fault_code") == config.FAULT_COORDINATOR]
    coordinator_status = "offline" if coord_faults else "online"

    total = len(config.NODE_IDS)
    health = 100.0 * (counts["ONLINE"] + 0.5 * counts["STALE"]) / total if total else 0.0
    if coordinator_status == "offline":
        health = 0.0
    health = max(0.0, health - 5.0 * len(active_faults))

    return _ok({
        "total_nodes": total,
        "online": counts["ONLINE"],
        "stale": counts["STALE"],
        "offline": counts["OFFLINE"],
        "unknown": counts["UNKNOWN"],
        "coordinator": coordinator_status,
        "active_fault_count": len(active_faults),
        "active_faults": active_faults,
        "network_health_pct": round(health, 1),
    })


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({"service": "HEALNET Local API", "docs": "/docs", "health": "/api/system/health"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="HEALNET Local REST API")
    ap.add_argument("--host", default=os.environ.get("HEALNET_API_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("HEALNET_API_PORT", "8000")))
    ap.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not _HAVE_FASTAPI:
        log.error("FastAPI or uvicorn not installed. Run: pip install fastapi uvicorn")
        return
    log.info("Starting HEALNET API on http://%s:%d", args.host, args.port)
    log.info("Interactive docs: http://%s:%d/docs", args.host, args.port)
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    main()

