"""
Full system test suite for HEALNET:
- ML pipeline & ModelManager
- Edge metrics & latency tracking
- Local REST API endpoints
- Coordinator events and recovery event storage
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from edge_metrics import EdgeMetrics, LatencyTracker
from storage import Storage
import protocol
import processor as P
from ml_pipeline import (
    _generate_normal_samples,
    _generate_abnormal_samples,
    generate_synthetic_dataset,
    engineer_features,
    time_aware_split,
    run_training_pipeline,
    predict,
    ModelManager,
)


class TestEdgeMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = EdgeMetrics()

    def test_latency_tracking(self):
        with self.metrics.track("test_op"):
            time.sleep(0.01)
        summary = self.metrics.latency_summary("test_op")
        self.assertEqual(summary["count"], 1)
        self.assertGreater(summary["min_ms"], 5.0)

    def test_counters(self):
        self.metrics.inc_messages(5)
        self.metrics.inc_faults(2)
        self.metrics.inc_anomalies(1)
        snap = self.metrics.snapshot()
        self.assertEqual(snap["messages_total"], 5)
        self.assertEqual(snap["faults_total"], 2)
        self.assertEqual(snap["anomalies_total"], 1)

    def test_snapshot_structure(self):
        snap = self.metrics.snapshot()
        for key in ("ts", "uptime_s", "uptime_human", "latencies", "os"):
            self.assertIn(key, snap)


class TestMLPipeline(unittest.TestCase):
    def test_data_generation_separation(self):
        import numpy as np
        rng = np.random.default_rng(42)
        normal = _generate_normal_samples(100, node_id=1, rng=rng)
        abnormal = _generate_abnormal_samples(20, node_id=1, rng=rng)
        self.assertEqual(len(normal), 100)
        self.assertEqual(len(abnormal), 20)
        self.assertEqual(normal.shape[1], 3)
        self.assertEqual(abnormal.shape[1], 3)

    def test_feature_engineering_and_split(self):
        X, y = generate_synthetic_dataset(n_normal=200, n_abnormal=30)
        X_feat = engineer_features(X)
        self.assertEqual(X_feat.shape[0], len(X))
        self.assertEqual(X_feat.shape[1], 15)
        X_tr, X_te, y_tr, y_te = time_aware_split(X_feat, y, 0.7)
        self.assertGreater(len(X_tr), 0)
        self.assertGreater(len(X_te), 0)
        self.assertEqual(len(X_tr) + len(X_te), len(X_feat))

    def test_model_training_and_metrics(self):
        res = run_training_pipeline(n_normal=300, n_abnormal=40)
        self.assertIn("all_results", res)
        self.assertIn("best_result", res)
        best = res["best_result"]
        self.assertIn("model_name", best)
        self.assertIn("precision", best)
        self.assertIn("recall", best)
        self.assertIn("f1", best)
        self.assertGreater(best["recall"], 0.7)

    def test_model_manager_and_predict(self):
        mgr = ModelManager()
        loaded = mgr.load()
        if loaded:
            is_anom, score = mgr.predict(26.0, 55.0, 1800, node_id=1)
            self.assertIsInstance(is_anom, bool)
            self.assertIsInstance(score, float)
            summary = mgr.summary()
            self.assertTrue(summary["loaded"])
            self.assertIn("model_type", summary)


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api import app
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        r = self.client.get("/api/system/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["data"]["status"], "healthy")

    def test_metrics_endpoint(self):
        r = self.client.get("/api/system/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("latencies", r.json()["data"])

    def test_sensors_endpoint(self):
        r = self.client.get("/api/sensors")
        self.assertEqual(r.status_code, 200)

    def test_nodes_endpoint(self):
        r = self.client.get("/api/nodes")
        self.assertEqual(r.status_code, 200)

    def test_faults_endpoint(self):
        r = self.client.get("/api/faults")
        self.assertEqual(r.status_code, 200)
        r_act = self.client.get("/api/faults/active")
        self.assertEqual(r_act.status_code, 200)

    def test_anomalies_endpoint(self):
        r = self.client.get("/api/anomalies")
        self.assertEqual(r.status_code, 200)

    def test_ml_status_and_metrics(self):
        r = self.client.get("/api/ml/status")
        self.assertEqual(r.status_code, 200)
        r_m = self.client.get("/api/ml/metrics")
        self.assertIn(r_m.status_code, (200, 404))

    def test_aliases_and_evaluation_endpoint(self):
        # Health alias
        r_health = self.client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)

        # Readings aliases
        r_readings = self.client.get("/api/readings")
        self.assertEqual(r_readings.status_code, 200)
        r_latest = self.client.get("/api/readings/latest")
        self.assertEqual(r_latest.status_code, 200)

        # Events alias
        r_events = self.client.get("/api/events")
        self.assertEqual(r_events.status_code, 200)

        # ML Evaluation endpoint
        r_eval = self.client.get("/api/ml/evaluation")
        self.assertIn(r_eval.status_code, (200, 404))


class TestProcessorIntegration(unittest.TestCase):
    def setUp(self):
        config.DB_PATH = os.path.join(tempfile.mkdtemp(), "t_proc_int.db")
        self.p = P.Processor()

    def test_fault_backend_lifecycle(self):
        # Simulate MQTT disconnect
        self.p._on_disconnect(None, None, None, "test_broker_down", None)
        self.assertIn(config.FAULT_BACKEND, self.p._sys.faults)
        self.assertFalse(self.p.mqtt_ok)

        # Active faults should contain FAULT 6
        active = self.p.db.active_faults()
        self.assertTrue(any(f["fault_code"] == config.FAULT_BACKEND for f in active))

        # Simulate reconnect
        dummy_flags = type("Flags", (), {"session_present": False})()
        dummy_rc = type("RC", (), {"is_failure": False})()
        self.p._on_connect(type("Client", (), {"subscribe": lambda *a, **k: None})(), None, dummy_flags, dummy_rc, None)
        self.assertTrue(self.p.mqtt_ok)
        self.assertNotIn(config.FAULT_BACKEND, self.p._sys.faults)

    def test_processor_metrics_sqlite_persistence(self):
        before = self.p.metrics.snapshot()
        self.p.metrics.inc_messages(10)
        self.p.metrics.inc_faults(3)
        self.p.publish_stats()
        loaded = self.p.db.load_system_metrics()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["messages_total"], before["messages_total"] + 10)
        self.assertEqual(loaded["faults_total"], before["faults_total"] + 3)


class TestSerialBridgeLock(unittest.TestCase):
    def test_lock_acquisition_and_release(self):
        from serial_bridge import _acquire_instance_lock, _release_instance_lock
        td = tempfile.mkdtemp()
        lock_file = os.path.join(td, "test_bridge.lock")

        # First acquisition should succeed
        ok, pid = _acquire_instance_lock(lock_file)
        self.assertTrue(ok)
        self.assertEqual(pid, os.getpid())

        # Second acquisition should fail because current PID is still running
        ok2, pid2 = _acquire_instance_lock(lock_file)
        self.assertFalse(ok2)
        self.assertEqual(pid2, os.getpid())

        # Release lock
        _release_instance_lock(lock_file)
        self.assertFalse(os.path.exists(lock_file))


class TestCoordinatorEventAndRecovery(unittest.TestCase):
    def setUp(self):
        config.DB_PATH = os.path.join(tempfile.mkdtemp(), "t_event.db")
        self.p = P.Processor()

    def test_coordinator_recovery_event_recorded(self):
        # Coordinator sends a manual recovery event (e.g. from physical push button)
        rec_event = {
            "type": "event",
            "event": "recovery_simulated",
            "node_id": 1,
            "message": "Node 1 software recovery triggered via coordinator button",
            "uptime_ms": 155476
        }
        self.p._on_message(None, None, type("Msg", (), {"topic": config.T_COORD_EVENT, "payload": str(rec_event).replace("'", '"').encode("utf-8")}))

        # Verify recovery event was recorded in recovery_events table
        recs = self.p.db.recovery_events()
        self.assertGreater(len(recs), 0)
        self.assertEqual(recs[0]["node_id"], 1)
        self.assertEqual(recs[0]["method"], "manual_button")

        # Verify system event was also logged
        sys_evs = self.p.db.system_events()
        self.assertTrue(any("Recovery on node 1" in str(e["detail"]) for e in sys_evs))

    def test_coordinator_failure_simulation_event_recorded(self):
        fail_event = {
            "type": "event",
            "event": "failure_simulated",
            "node_id": 2,
            "message": "Node 2 communication failure simulated via coordinator button",
            "uptime_ms": 155476
        }
        self.p._on_message(None, None, type("Msg", (), {"topic": config.T_COORD_EVENT, "payload": str(fail_event).replace("'", '"').encode("utf-8")}))

        sys_evs = self.p.db.system_events()
        self.assertTrue(any("Failure simulation on node 2" in str(e["detail"]) for e in sys_evs))


if __name__ == "__main__":
    unittest.main()
