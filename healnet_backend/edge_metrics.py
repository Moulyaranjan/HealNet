"""
HEALNET edge-computing metrics collector.

Monitors local resource usage (CPU, memory, disk) and application-level
latencies (MQTT message processing, ML inference, DB writes).

All monitoring is lightweight:
- Uses psutil for OS metrics (optional; gracefully degrades if absent).
- Thread-safe rolling windows for latency tracking.
- No external network calls; all data stays local.

Usage:
    metrics = EdgeMetrics()
    # In message handler:
    with metrics.track("mqtt_processing"):
        process_message(...)
    # In API handler:
    print(metrics.snapshot())
"""
import contextlib
import logging
import threading
import time
from collections import deque
from typing import Optional

log = logging.getLogger("healnet.edge_metrics")

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False
    log.warning("psutil not installed – OS resource monitoring disabled. pip install psutil")


class LatencyTracker:
    """Rolling window of operation latencies in milliseconds."""

    def __init__(self, name: str, maxlen: int = 200):
        self.name = name
        self._lock = threading.Lock()
        self._samples: deque[float] = deque(maxlen=maxlen)
        self._total_count = 0
        self._error_count = 0

    def record(self, latency_ms: float, error: bool = False):
        with self._lock:
            self._samples.append(latency_ms)
            self._total_count += 1
            if error:
                self._error_count += 1

    def summary(self) -> dict:
        with self._lock:
            samples = list(self._samples)
            total = self._total_count
            errors = self._error_count
        if not samples:
            return {
                "name": self.name,
                "count": total,
                "errors": errors,
                "min_ms": None,
                "max_ms": None,
                "mean_ms": None,
                "p95_ms": None,
            }
        samples_sorted = sorted(samples)
        n = len(samples_sorted)
        p95_idx = max(0, int(n * 0.95) - 1)
        return {
            "name": self.name,
            "count": total,
            "errors": errors,
            "min_ms": round(samples_sorted[0], 2),
            "max_ms": round(samples_sorted[-1], 2),
            "mean_ms": round(sum(samples_sorted) / n, 2),
            "p95_ms": round(samples_sorted[p95_idx], 2),
        }


class EdgeMetrics:
    """
    Collects and exposes edge-computing metrics for HEALNET.

    Latency categories tracked:
        - mqtt_processing  : time from MQTT on_message callback to end of processing
        - db_write         : time for SQLite insert_reading
        - ml_inference     : time for ML model predict()
        - fault_detection  : time for deterministic fault checks
    """

    def __init__(self):
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._message_count = 0
        self._fault_count = 0
        self._anomaly_count = 0

        self.trackers: dict[str, LatencyTracker] = {
            "mqtt_processing": LatencyTracker("mqtt_processing"),
            "db_write": LatencyTracker("db_write"),
            "ml_inference": LatencyTracker("ml_inference"),
            "fault_detection": LatencyTracker("fault_detection"),
        }

    # ------------------------------------------------------------------
    # Context manager for automatic latency tracking
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def track(self, operation: str):
        """
        Usage:
            with metrics.track("db_write"):
                db.insert_reading(row)
        """
        if operation not in self.trackers:
            self.trackers[operation] = LatencyTracker(operation)
        t0 = time.perf_counter()
        error = False
        try:
            yield
        except Exception:
            error = True
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.trackers[operation].record(elapsed_ms, error=error)

    def record_latency(self, operation: str, latency_ms: float, error: bool = False):
        """Direct latency recording without context manager."""
        if operation not in self.trackers:
            self.trackers[operation] = LatencyTracker(operation)
        self.trackers[operation].record(latency_ms, error)

    # ------------------------------------------------------------------
    # Event counters
    # ------------------------------------------------------------------
    def inc_messages(self, n: int = 1):
        with self._lock:
            self._message_count += n

    def inc_faults(self, n: int = 1):
        with self._lock:
            self._fault_count += n

    def inc_anomalies(self, n: int = 1):
        with self._lock:
            self._anomaly_count += n

    # ------------------------------------------------------------------
    # OS resource monitoring
    # ------------------------------------------------------------------
    def _os_metrics(self) -> dict:
        if not _HAVE_PSUTIL:
            return {"available": False, "note": "psutil not installed"}
        try:
            proc = psutil.Process()
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            proc_mem = proc.memory_info()
            disk = psutil.disk_usage(".")
            return {
                "available": True,
                "cpu_percent": cpu_pct,
                "memory_total_mb": round(mem.total / 1e6, 1),
                "memory_used_mb": round(mem.used / 1e6, 1),
                "memory_available_mb": round(mem.available / 1e6, 1),
                "memory_percent": mem.percent,
                "process_rss_mb": round(proc_mem.rss / 1e6, 2),
                "process_vms_mb": round(proc_mem.vms / 1e6, 2),
                "disk_total_gb": round(disk.total / 1e9, 1),
                "disk_used_gb": round(disk.used / 1e9, 1),
                "disk_free_gb": round(disk.free / 1e9, 1),
                "disk_percent": disk.percent,
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return a complete metrics snapshot as a serialisable dict."""
        with self._lock:
            msg_count = self._message_count
            fault_count = self._fault_count
            anomaly_count = self._anomaly_count

        uptime_s = time.time() - self._start_time
        return {
            "ts": round(time.time(), 3),
            "uptime_s": round(uptime_s, 1),
            "uptime_human": _fmt_uptime(uptime_s),
            "messages_total": msg_count,
            "faults_total": fault_count,
            "anomalies_total": anomaly_count,
            "throughput_msg_per_min": round(msg_count / max(uptime_s / 60, 0.001), 2),
            "latencies": {k: v.summary() for k, v in self.trackers.items()},
            "os": self._os_metrics(),
        }

    def latency_summary(self, operation: Optional[str] = None) -> dict:
        """Return latency summary for one or all operations."""
        if operation:
            if operation in self.trackers:
                return self.trackers[operation].summary()
            return {}
        return {k: v.summary() for k, v in self.trackers.items()}


def _fmt_uptime(seconds: float) -> str:
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# Module-level singleton used by processor and API
_default_metrics: Optional[EdgeMetrics] = None


def get_metrics() -> EdgeMetrics:
    """Return the shared EdgeMetrics instance (created on first call)."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = EdgeMetrics()
    return _default_metrics

