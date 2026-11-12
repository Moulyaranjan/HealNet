"""
Comprehensive validation unit tests for HEALNET processor and protocol:
- Strict boolean vs integer node_id and sequence rejection
- String boolean ("false", "true") parsing
- Negative and invalid hop counts
- Self-forwarding rejection
- via and forwarded_by consistency
- Coordinator recovery event distinction (initiation vs confirmation)
- Coordinator disconnect marking nodes UNKNOWN and sensors MISSING
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import processor as P
import protocol


class ValidationCapture(P.Processor):
    def __init__(self, db_path=None):
        config.DB_PATH = db_path or os.path.join(tempfile.mkdtemp(), "test_val.db")
        config.STARTUP_GRACE_S = 0
        super().__init__()
        self.out = []

    def pub(self, topic, payload, qos=0, retain=False):
        self.out.append((topic, payload))


class TestStrictValidation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "val_test.db")
        self.p = ValidationCapture(self.db_path)
        self.p.coordinator_online = True

    def tearDown(self):
        self.p.db.close()

    def test_reject_boolean_node_id(self):
        # node=True should be rejected (must be int, not bool)
        clean, reason = self.p._validate("healnet/node1/data", {"node": True, "seq": 1})
        self.assertIsNone(clean)
        self.assertIn("node id must be an integer", reason)

    def test_reject_boolean_seq(self):
        # seq=True should be rejected
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": True})
        self.assertIsNone(clean)
        self.assertIn("missing/invalid seq", reason)

    def test_reject_negative_seq(self):
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": -1})
        self.assertIsNone(clean)
        self.assertIn("missing/invalid seq", reason)

    def test_reject_self_forwarding(self):
        # Node 1 forwarding for Node 1
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": 1, "via": 1, "hops": 1})
        self.assertIsNone(clean)
        self.assertIn("self-forwarding invalid", reason)

    def test_reject_invalid_via_node(self):
        # via=99 not in NODE_IDS
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": 1, "via": 99, "hops": 1})
        self.assertIsNone(clean)
        self.assertIn("not a valid node id", reason)

    def test_reject_negative_hops(self):
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": 1, "via": 2, "hops": -1})
        self.assertIsNone(clean)
        self.assertIn("hops must be non-negative integer", reason)

    def test_reject_direct_with_nonzero_hops(self):
        # via is None but hops = 2
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": 1, "via": None, "hops": 2})
        self.assertIsNone(clean)
        self.assertIn("direct packet (via is null) must have hops=0", reason)

    def test_reject_forwarded_with_zero_hops(self):
        # via is 2 but hops = 0
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": 1, "via": 2, "hops": 0})
        self.assertIsNone(clean)
        self.assertIn("forwarded packet (via is specified) must have hops >= 1", reason)

    def test_reject_inconsistent_forwarded_by(self):
        # via is 2 but forwarded_by says NODE1
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": 1, "via": 2, "hops": 1, "forwarded_by": "NODE1"})
        self.assertIsNone(clean)
        self.assertIn("inconsistent with via", reason)

    def test_parse_string_booleans(self):
        # valid="false" must parse as False, not True
        clean, reason = self.p._validate("healnet/node1/data", {"node": 1, "seq": 1, "temp": 25.0, "hum": 50.0, "light": 1000, "valid": "false"})
        self.assertIsNotNone(clean)
        self.assertFalse(clean["valid"])

        # valid="true" must parse as True
        clean2, _ = self.p._validate("healnet/node1/data", {"node": 1, "seq": 2, "temp": 25.0, "hum": 50.0, "light": 1000, "valid": "true"})
        self.assertIsNotNone(clean2)
        self.assertTrue(clean2["valid"])

    def test_coordinator_event_does_not_confirm_recovery(self):
        # Sending a coordinator recovery notification records INITIATED, not SUCCESS
        self.p._on_coordinator_event({
            "event": "manual_recovery",
            "node_id": 1,
            "message": "Button pressed"
        })
        recs = self.p.db.recovery_events(limit=5)
        self.assertGreaterEqual(len(recs), 1)
        self.assertEqual(recs[0]["result"], "INITIATED")
        self.assertNotEqual(recs[0]["result"], "SUCCESS")

    def test_coordinator_disconnect_marks_nodes_unknown_and_missing(self):
        now = time.time()
        # Initial direct packet from node 1
        self.p._on_data("healnet/node1/data", {
            "node": 1, "seq": 1, "temp": 25.0, "hum": 50.0, "light": 1500, "rx_ts": now
        })
        st1 = self.p.nodes[1]
        self.assertEqual(st1.esp32_status, config.STATUS_ONLINE)
        self.assertEqual(st1.sensor_status, config.SENSOR_AVAILABLE)

        # Coordinator disconnects
        self.p._on_coordinator({"status": "offline"})
        self.p.watchdog_tick()

        # Both node hardware status and sensor status must reflect unobservable / missing
        self.assertEqual(st1.esp32_status, config.STATUS_UNKNOWN)
        self.assertEqual(st1.sensor_status, config.SENSOR_UNAVAILABLE)
        self.assertEqual(st1.data_status, config.DATA_MISSING)


if __name__ == "__main__":
    unittest.main()

