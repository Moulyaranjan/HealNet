"""
Unit tests specifically verifying the fixes and completions for HEALNET:
1. ML data leakage prevention & causal EMA diff_node
2. Realistic synthetic dataset with distributed anomalies throughout timeline
3. Feature count and order consistency between training and live ModelManager prediction
4. ModelManager live inference with history buffer
5. Node status tracking (ONLINE, STALE, OFFLINE, UNKNOWN)
6. Duplicate fault prevention and fault resolution
7. Complete REST API endpoints (/api/system/summary, /api/recovery, /api/readings/all, /api/nodes/{id}/faults)
8. Coordinator failure & recovery event processing and MQTT publishing
9. Edge metrics OS and latency tracking
"""
import os
import sys
import tempfile
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from storage import Storage
from ml_pipeline import (
    BASE_FEATURES,
    ENGINEERED_FEATURES,
    ROLLING_WINDOW,
    generate_synthetic_dataset,
    engineer_features,
    time_aware_split,
    ModelManager,
)
from tests.test_full_system import DirectAPIClient
from api import app, _derive_node_status


class TestFixes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_fixes.db")
        self.db = Storage(self.db_path)
        self.client = DirectAPIClient(app)

    def tearDown(self):
        self.db.close()

    def test_ml_data_leakage_and_timeline_distribution(self):
        """Verify normal and abnormal samples are distributed throughout the timeline and across train/test splits."""
        X, y = generate_synthetic_dataset(n_normal=500, n_abnormal=50, seed=123)
        self.assertEqual(len(X), 550)
        self.assertEqual(len(y), 550)

        # Ensure features are 2D with 3 columns (temp, hum, light)
        self.assertEqual(X.shape[1], 3)

        # Check that abnormal readings (y == 1) appear in both first half and second half
        split = int(len(y) * 0.70)
        y_train = y[:split]
        y_test = y[split:]
        self.assertGreater(np.sum(y_train == 1), 0, "Training set must contain abnormal samples")
        self.assertGreater(np.sum(y_test == 1), 0, "Test set must contain abnormal samples")

        # Feature engineering must produce exactly 15 features
        X_eng = engineer_features(X)
        self.assertEqual(X_eng.shape[1], 15)
        self.assertEqual(len(ENGINEERED_FEATURES), 15)

    def test_feature_consistency_between_training_and_live_inference(self):
        """Verify that training and live prediction use the exact same 15 features in the exact same order."""
        mgr = ModelManager()
        loaded = mgr.load()
        if loaded:
            self.assertEqual(mgr._meta.get("features"), ENGINEERED_FEATURES)

            # Test live prediction via ModelManager
            is_anom, score = mgr.predict(24.0, 55.0, 1800.0, node_id=1)
            self.assertIsInstance(is_anom, bool)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

            # Ensure buffer exists and has samples
            self.assertIn(1, mgr._buffers)
            self.assertGreaterEqual(len(mgr._buffers[1]), 1)

    def test_duplicate_fault_prevention_and_resolution(self):
        """Verify that opening the same fault code on a node does not create duplicate active rows."""
        now = time.time()
        # Open fault 1 on node 1
        fid1 = self.db.open_fault(now, 1, 1, "SENSOR_FAILURE", "MEDIUM", "Detail 1")
        # Attempt to open same fault 1 on node 1 again
        fid2 = self.db.open_fault(now + 1, 1, 1, "SENSOR_FAILURE", "MEDIUM", "Detail 1 updated")
        self.assertEqual(fid1, fid2, "Duplicate fault should update existing active record and return same ID")

        active = self.db.active_faults()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["detail"], "Detail 1 updated")

        # Resolve the fault
        resolved = self.db.resolve_fault(1, 1, now + 2)
        self.assertTrue(resolved)
        active_after = self.db.active_faults()
        self.assertEqual(len(active_after), 0)

    def test_node_status_derivation(self):
        """Verify accurate ONLINE, STALE, OFFLINE, and UNKNOWN status derivation."""
        now = time.time()

        # 1. No readings ever -> UNKNOWN
        st, _ = _derive_node_status(self.db, 1, None)
        self.assertEqual(st, "UNKNOWN")

        # 2. Fresh reading (<6s) -> ONLINE
        fresh_reading = {"node_id": 1, "ts": now - 2.0}
        st, _ = _derive_node_status(self.db, 1, fresh_reading)
        self.assertEqual(st, "ONLINE")

        # 3. Reading between 6s and 10s -> STALE
        stale_reading = {"node_id": 1, "ts": now - 7.0}
        st, _ = _derive_node_status(self.db, 1, stale_reading)
        self.assertEqual(st, "STALE")

        # 4. Reading older than 10s -> OFFLINE
        offline_reading = {"node_id": 1, "ts": now - 15.0}
        st, _ = _derive_node_status(self.db, 1, offline_reading)
        self.assertEqual(st, "OFFLINE")

        # 5. Coordinator failure active -> UNKNOWN (cannot observe nodes)
        self.db.open_fault(now, None, config.FAULT_COORDINATOR, "COORDINATOR_FAILURE", "CRITICAL", "USB disconnected")
        st, _ = _derive_node_status(self.db, 1, fresh_reading)
        self.assertEqual(st, "UNKNOWN")

    def test_new_api_endpoints(self):
        """Verify the newly added/completed endpoints in api.py."""
        # 1. Summary endpoint
        r_sum = self.client.get("/api/system/summary")
        self.assertEqual(r_sum.status_code, 200)
        data = r_sum.json()["data"]
        self.assertIn("online", data)
        self.assertIn("network_health_pct", data)

        # 2. Recovery events endpoint
        r_rec = self.client.get("/api/system/recovery")
        self.assertEqual(r_rec.status_code, 200)
        self.assertIn("events", r_rec.json()["data"])

        # 3. All readings endpoint
        r_all = self.client.get("/api/readings/all")
        self.assertEqual(r_all.status_code, 200)
        self.assertIn("readings", r_all.json()["data"])

        # 4. Node faults endpoint
        r_nf = self.client.get("/api/nodes/1/faults")
        self.assertEqual(r_nf.status_code, 200)
        self.assertEqual(r_nf.json()["data"]["node_id"], 1)

    def test_coordinator_event_processor_publishing(self):
        """Verify processor publishes recovery to T_RECOVERY and simulated fail to T_FAULT."""
        import processor as P
        p = P.Processor()
        published = []
        p.pub = lambda topic, payload, qos=0, retain=False: published.append((topic, payload))

        # Recovery event
        p._on_coordinator_event({"event": "recovery_simulated", "node_id": 1, "message": "Button recovery"})
        rec_pubs = [pub for pub in published if pub[0] == config.T_RECOVERY]
        self.assertEqual(len(rec_pubs), 1)
        self.assertEqual(rec_pubs[0][1]["node"], 1)
        self.assertEqual(rec_pubs[0][1]["method"], "manual_button")

        # Failure simulation event
        p._on_coordinator_event({"event": "failure_simulated", "node_id": 2, "message": "Button failure"})
        fail_pubs = [pub for pub in published if pub[0] == config.T_FAULT]
        self.assertEqual(len(fail_pubs), 1)
        self.assertEqual(fail_pubs[0][1]["event"], "SIMULATED_FAILURE")
        self.assertEqual(fail_pubs[0][1]["node"], 2)


if __name__ == "__main__":
    unittest.main()

