"""
Test suite validating the 8 Project Advancement Scenarios for HEALNET:
TEST 1: Normal Operation
TEST 2: Node 1 Failure
TEST 3: Node 1 Sensor Failover via CD4053
TEST 4: Invalid Sensor Reading & Cache Protection
TEST 5: Node 1 Recovery Verification via Direct Packets
TEST 6: Coordinator Failure & Node Isolation
TEST 7: Duplicate and Out-of-Order Packets
TEST 8: ML Anomaly Detection with Route Change Immunity
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
from storage import Storage


class AdvCapture(P.Processor):
    def __init__(self, db_path=None):
        config.DB_PATH = db_path or os.path.join(tempfile.mkdtemp(), "test_adv.db")
        config.STARTUP_GRACE_S = 0
        super().__init__()
        self.out = []

    def pub(self, topic, payload, qos=0, retain=False):
        self.out.append((topic, payload))

    def topics(self, t):
        return [p for (x, p) in self.out if x == t]


def make_packet(node, seq, temp=27.5, hum=55.0, light=1850, valid=True, via=None, hops=0, forwarded_by=None, source=None, rx_ts=None, **extra):
    pkt = {
        "type": "data",
        "node": node,
        "seq": seq,
        "temp": temp,
        "hum": hum,
        "light": light,
        "valid": valid,
        "via": via,
        "hops": hops,
        "source": source or f"NODE{node}",
        "forwarded_by": forwarded_by,
        "uptime_ms": seq * 2000,
        "rx_ts": rx_ts or time.time(),
    }
    pkt.update(extra)
    return pkt


class TestAdvancements(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "adv_test.db")
        self.p = AdvCapture(self.db_path)
        self.p.coordinator_online = True

    def tearDown(self):
        self.p.db.close()

    def feed(self, node, seq, **kwargs):
        pkt = make_packet(node, seq, **kwargs)
        self.p._on_data(config.t_data(node), pkt)

    # --------------------------------------------------------------------------
    # TEST 1: Normal Operation
    # Both Node 1 and Node 2 send data directly to the Coordinator.
    # --------------------------------------------------------------------------
    def test_01_normal_operation(self):
        now = time.time()
        self.feed(1, 1, temp=29.2, hum=54.1, light=1741, via=None, hops=0, rx_ts=now)
        self.feed(2, 1, temp=28.6, hum=57.3, light=965, via=None, hops=0, rx_ts=now)

        st1 = self.p.nodes[1]
        st2 = self.p.nodes[2]

        # Both nodes ONLINE
        self.assertEqual(st1.esp32_status, config.STATUS_ONLINE)
        self.assertEqual(st2.esp32_status, config.STATUS_ONLINE)

        # Both sensors AVAILABLE
        self.assertEqual(st1.sensor_status, config.SENSOR_AVAILABLE)
        self.assertEqual(st2.sensor_status, config.SENSOR_AVAILABLE)

        # Both routes DIRECT
        self.assertEqual(st1.route_status, config.ROUTE_DIRECT)
        self.assertEqual(st2.route_status, config.ROUTE_DIRECT)
        self.assertIsNone(st1.route[0])
        self.assertEqual(st1.route[1], 0)

        # Correct sensor readings stored in SQLite
        row1 = self.p.db.latest_reading(1)
        row2 = self.p.db.latest_reading(2)
        self.assertIsNotNone(row1)
        self.assertIsNotNone(row2)
        self.assertEqual(row1["node_id"], 1)
        self.assertAlmostEqual(row1["temp"], 29.2, places=1)
        self.assertEqual(row2["node_id"], 2)
        self.assertAlmostEqual(row2["temp"], 28.6, places=1)

    # --------------------------------------------------------------------------
    # TEST 2: Node 1 Failure
    # Node 1 ESP32 stops communicating.
    # --------------------------------------------------------------------------
    def test_02_node1_failure(self):
        now = time.time()
        # Initial direct packet from both nodes
        self.feed(1, 1, rx_ts=now)
        self.feed(2, 1, rx_ts=now)

        # Simulate Node 1 stopping direct communication for > 10s
        self.p.nodes[1].last_rx_ts = now - 15.0
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.watchdog_tick()

        st1 = self.p.nodes[1]
        st2 = self.p.nodes[2]

        # Node 1 ESP32 becomes OFFLINE
        self.assertEqual(st1.esp32_status, config.STATUS_OFFLINE)
        self.assertEqual(st1.status, P.OFFLINE)

        # Communication fault is recorded
        self.assertIn(config.FAULT_NODE_COMM, st1.faults)
        active_faults = self.p.db.active_faults()
        n1_faults = [f for f in active_faults if f["node_id"] == 1 and f["fault_code"] == config.FAULT_NODE_COMM]
        self.assertEqual(len(n1_faults), 1)

        # Failover initiated state is set
        self.assertEqual(st1.failover_state, config.STATE_FAILOVER_INITIATED)

        # Node 2 continues normal operation
        self.assertEqual(st2.esp32_status, config.STATUS_ONLINE)
        self.assertEqual(st2.sensor_status, config.SENSOR_AVAILABLE)

    # --------------------------------------------------------------------------
    # TEST 3: Node 1 Sensor Failover
    # Node 2 accesses Node 1 physical sensors through CD4053 and forwards them.
    # --------------------------------------------------------------------------
    def test_03_node1_sensor_failover(self):
        now = time.time()
        # Node 1 direct communication had timed out
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE

        # Node 2 forwards Node 1 sensor readings via CD4053: node=1, via=2, hops=1
        self.feed(
            1, 10,
            temp=29.2, hum=54.1, light=1725, valid=True,
            via=2, hops=1, forwarded_by="NODE2", source="NODE1", rx_ts=now
        )

        st1 = self.p.nodes[1]

        # Node 1 data is still identified as Node 1 data
        self.assertEqual(st1.node_id, 1)

        # Node 1 ESP32 remains OFFLINE (forwarded packet does not spoof ESP32 liveness)
        self.assertEqual(st1.esp32_status, config.STATUS_OFFLINE)

        # Node 1 physical sensor availability becomes AVAILABLE!
        self.assertEqual(st1.sensor_status, config.SENSOR_AVAILABLE)

        # Sensor data is FRESH
        self.assertEqual(st1.data_status, config.DATA_FRESH)

        # Route becomes FAILOVER VIA NODE 2
        self.assertEqual(st1.route_status, config.ROUTE_FAILOVER)
        self.assertEqual(st1.route, (2, 1))

        # Failover active state is recorded
        self.assertEqual(st1.failover_state, config.STATE_FAILOVER_ACTIVE)

        # Data is stored under Node 1 in SQLite
        row = self.p.db.latest_reading(1)
        self.assertEqual(row["node_id"], 1)
        self.assertEqual(row["via"], 2)
        self.assertEqual(row["hops"], 1)
        self.assertAlmostEqual(row["temp"], 29.2, places=1)

    # --------------------------------------------------------------------------
    # TEST 4: Invalid Sensor Reading & Cache Protection
    # Send invalid or missing sensor values.
    # --------------------------------------------------------------------------
    def test_04_invalid_sensor_reading(self):
        # 1. Establish valid readings first
        self.feed(1, 1, temp=25.0, hum=50.0, light=1500, valid=True)
        st1 = self.p.nodes[1]
        self.assertEqual(st1.last_valid_readings["temp"], 25.0)

        # 2. Send corrupted / invalid readings
        for s in range(2, 6):
            self.feed(1, s, temp=None, hum=None, valid=False, dht_ok=False)

        # Invalid readings identified
        self.assertEqual(st1.data_status, config.DATA_INVALID)
        self.assertIn(config.FAULT_SENSOR, st1.faults)

        # Previous valid readings are NOT silently overwritten with null
        self.assertEqual(st1.last_valid_readings["temp"], 25.0)
        self.assertEqual(st1.last_valid_readings["hum"], 50.0)

    # --------------------------------------------------------------------------
    # TEST 5: Node 1 Recovery — Two-Packet Validation
    # Recovery must NOT happen after one direct packet.
    # Recovery MUST happen after two consecutive valid direct packets.
    # --------------------------------------------------------------------------
    def test_05_recovery_not_after_one_packet(self):
        """Recovery does NOT happen after only one direct packet."""
        now = time.time()
        # Establish failover state
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]
        self.assertEqual(st1.route_status, config.ROUTE_FAILOVER)
        self.assertEqual(st1.failover_state, config.STATE_FAILOVER_ACTIVE)

        # First valid direct packet: starts validation but does NOT confirm
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        self.assertEqual(st1.failover_state, config.STATE_RECOVERY_VALIDATION)
        self.assertEqual(st1.recovery_validation_count, 1)
        # Route still FAILOVER!
        self.assertEqual(st1.route_status, config.ROUTE_FAILOVER)
        # Fault still active!
        self.assertIn(config.FAULT_NODE_COMM, st1.faults)

    def test_05b_recovery_after_two_packets(self):
        """Recovery IS confirmed after two consecutive valid direct packets."""
        now = time.time()
        # Establish failover state
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]
        self.assertEqual(st1.failover_state, config.STATE_FAILOVER_ACTIVE)

        # First valid direct packet
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        self.assertEqual(st1.failover_state, config.STATE_RECOVERY_VALIDATION)
        self.assertEqual(st1.recovery_validation_count, 1)

        # Second valid direct packet — recovery confirmed!
        self.feed(1, 13, temp=29.1, hum=53.2, light=1710, via=None, hops=0, rx_ts=now + 2)
        self.assertEqual(st1.esp32_status, config.STATUS_ONLINE)
        self.assertEqual(st1.route_status, config.ROUTE_DIRECT)
        self.assertEqual(st1.route, (None, 0))
        self.assertEqual(st1.failover_state, config.STATE_NORMAL)
        self.assertNotIn(config.FAULT_NODE_COMM, st1.faults)

        # Recovery event stored in database
        rec_events = self.p.db.recovery_events(limit=5)
        self.assertGreaterEqual(len(rec_events), 1)
        r = rec_events[0]
        self.assertEqual(r["node_id"], 1)
        self.assertEqual(r["result"], "SUCCESS")
        self.assertEqual(r["new_route"], config.ROUTE_DIRECT)

    def test_05c_forwarded_packets_dont_count(self):
        """Forwarded packets NEVER count towards recovery validation."""
        now = time.time()
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]

        # First valid direct packet
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        self.assertEqual(st1.recovery_validation_count, 1)

        # Forwarded packet arrives — must NOT count and must reset validation
        self.feed(1, 13, via=2, hops=1, forwarded_by="NODE2", rx_ts=now + 2)
        self.assertEqual(st1.recovery_validation_count, 0)
        self.assertEqual(st1.route_status, config.ROUTE_FAILOVER)
        self.assertNotEqual(st1.failover_state, config.STATE_NORMAL)

    def test_05d_invalid_direct_packets_dont_count(self):
        """Invalid sensor data in direct packets does NOT count towards recovery."""
        now = time.time()
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]

        # First valid direct packet
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        self.assertEqual(st1.recovery_validation_count, 1)

        # Invalid direct packet — resets counter
        self.feed(1, 13, temp=None, hum=None, light=None, valid=False, dht_ok=False,
                  via=None, hops=0, rx_ts=now + 2)
        self.assertEqual(st1.recovery_validation_count, 0)
        self.assertEqual(st1.route_status, config.ROUTE_FAILOVER)

    def test_05e_duplicate_packets_dont_count(self):
        """Duplicate packets are dropped entirely and do NOT count."""
        now = time.time()
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]

        # First valid direct packet
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        self.assertEqual(st1.recovery_validation_count, 1)

        # Duplicate of packet 12 — dropped by dedup, counter stays at 1
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now + 1)
        self.assertEqual(st1.recovery_validation_count, 1)
        self.assertEqual(st1.route_status, config.ROUTE_FAILOVER)

    def test_05f_ooo_packets_dont_count(self):
        """Out-of-order packets must NOT count towards recovery."""
        now = time.time()
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]

        # First valid direct packet (seq 14)
        self.feed(1, 14, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        self.assertEqual(st1.recovery_validation_count, 1)

        # Out-of-order packet (seq 12, older than 14) — resets counter
        self.feed(1, 12, temp=28.0, hum=52.0, light=1600, via=None, hops=0, rx_ts=now + 2)
        self.assertEqual(st1.recovery_validation_count, 0)
        self.assertEqual(st1.route_status, config.ROUTE_FAILOVER)

    def test_05g_interrupted_validation_resets(self):
        """An interruption (forwarded packet) during validation resets the counter."""
        now = time.time()
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]

        # First valid direct packet
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        self.assertEqual(st1.recovery_validation_count, 1)
        self.assertEqual(st1.failover_state, config.STATE_RECOVERY_VALIDATION)

        # Forwarded packet interrupts — resets counter
        self.feed(1, 13, via=2, hops=1, forwarded_by="NODE2", rx_ts=now + 2)
        self.assertEqual(st1.recovery_validation_count, 0)

        # Now start again: two valid consecutive direct packets
        self.feed(1, 14, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now + 4)
        self.assertEqual(st1.recovery_validation_count, 1)
        self.feed(1, 15, temp=29.1, hum=53.1, light=1710, via=None, hops=0, rx_ts=now + 6)
        # NOW recovery is confirmed
        self.assertEqual(st1.failover_state, config.STATE_NORMAL)
        self.assertEqual(st1.route_status, config.ROUTE_DIRECT)
        self.assertNotIn(config.FAULT_NODE_COMM, st1.faults)

    def test_05h_failover_active_during_validation(self):
        """Route remains FAILOVER and fault active during the entire validation phase."""
        now = time.time()
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.p.nodes[1].faults[config.FAULT_NODE_COMM] = {"id": 99, "detail": "test", "ts": now - 10}
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]

        # First valid direct packet — validation started
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        # Verify ALL these remain during validation
        self.assertEqual(st1.route_status, config.ROUTE_FAILOVER)
        self.assertIn(config.FAULT_NODE_COMM, st1.faults)
        self.assertIn(st1.failover_state, (config.STATE_RECOVERY_VALIDATION,))

    def test_05i_recovery_events_stored_correctly(self):
        """Recovery events in SQLite have the correct metadata."""
        now = time.time()
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]

        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        self.feed(1, 13, temp=29.1, hum=53.2, light=1710, via=None, hops=0, rx_ts=now + 2)

        rec_events = self.p.db.recovery_events(limit=5)
        self.assertGreaterEqual(len(rec_events), 1)
        r = rec_events[0]
        self.assertEqual(r["node_id"], 1)
        self.assertEqual(r["result"], "SUCCESS")
        self.assertEqual(r["new_route"], config.ROUTE_DIRECT)
        self.assertEqual(r["method"], "automatic_direct_verification")
        self.assertIsNotNone(r["start_ts"])

    def test_05j_route_changes_only_after_confirmation(self):
        """Route to DIRECT happens ONLY after the second valid direct packet."""
        now = time.time()
        self.p.nodes[1].last_direct_rx_ts = now - 15.0
        self.p.nodes[1].esp32_status = config.STATUS_OFFLINE
        self.p.nodes[1].status = P.OFFLINE
        self.feed(1, 10, via=2, hops=1, forwarded_by="NODE2", rx_ts=now - 5)
        st1 = self.p.nodes[1]

        # After first direct packet, route still FAILOVER
        self.feed(1, 12, temp=29.0, hum=53.0, light=1700, via=None, hops=0, rx_ts=now)
        route_events_before = self.p.topics(config.T_ROUTING)
        # Only the failover route event(s) should exist, no DIRECT route event yet
        direct_events = [e for e in route_events_before if e.get("state") == config.ROUTE_DIRECT]
        self.assertEqual(len(direct_events), 0)

        # After second direct packet, route becomes DIRECT
        self.feed(1, 13, temp=29.1, hum=53.2, light=1710, via=None, hops=0, rx_ts=now + 2)
        self.assertEqual(st1.route_status, config.ROUTE_DIRECT)

    # --------------------------------------------------------------------------
    # TEST 6: Coordinator Failure
    # Simulate Coordinator disconnection from backend.
    # --------------------------------------------------------------------------
    def test_06_coordinator_failure(self):
        self.feed(1, 1)
        self.feed(2, 1)

        # Coordinator disconnects
        self.p._on_coordinator({"status": "offline", "port": "COM14", "last_line_age_s": 8.0})
        self.assertIn(config.FAULT_COORDINATOR, self.p._system_state().faults)

        # Watchdog ticks after timeout
        now = time.time()
        for n in (1, 2):
            self.p.nodes[n].last_rx_ts = now - 20
        self.p.watchdog_tick()

        # Both nodes marked UNKNOWN (not failed individually), data marked MISSING
        for n in (1, 2):
            self.assertEqual(self.p.nodes[n].status, P.UNKNOWN)
            self.assertEqual(self.p.nodes[n].esp32_status, config.STATUS_UNKNOWN)

        # Coordinator reconnects
        self.p._on_coordinator({"status": "online"})
        self.assertNotIn(config.FAULT_COORDINATOR, self.p._system_state().faults)

    # --------------------------------------------------------------------------
    # TEST 7: Duplicate and Out-of-Order Packets
    # --------------------------------------------------------------------------
    def test_07_duplicates_and_out_of_order(self):
        # Packet 1 arrives
        self.feed(1, 10, temp=26.0)
        st = self.p.nodes[1]
        self.assertEqual(st.packets, 1)

        # Duplicate packet 10 arrives
        self.feed(1, 10, temp=26.0)
        self.assertEqual(st.dup, 1)
        self.assertEqual(st.packets, 1)  # packets counter not incremented on dup

        # Packet 12 arrives (skip 11 -> gap of 1)
        self.feed(1, 12, temp=26.5)
        self.assertEqual(st.lost, 1)

        # Late packet 11 arrives (out of order)
        self.feed(1, 11, temp=26.2)
        self.assertEqual(st.ooo, 1)
        # Old packet does not roll back last_seq
        self.assertEqual(st.last_seq, 12)

    # --------------------------------------------------------------------------
    # TEST 8: ML Anomaly Detection & Route Change Immunity
    # Route changes do NOT trigger sensor anomalies.
    # --------------------------------------------------------------------------
    def test_08_ml_anomaly_and_route_change_immunity(self):
        now = time.time()
        # Seed 15 normal readings
        for s in range(1, 16):
            self.feed(1, s, temp=24.0, hum=50.0, light=1800, via=None, hops=0, rx_ts=now + s * 2)

        # Route transitions to FAILOVER (Node 2 forwarding same normal readings)
        self.feed(1, 16, temp=24.0, hum=50.0, light=1800, via=2, hops=1, rx_ts=now + 32)
        st = self.p.nodes[1]

        # Route changed to failover
        self.assertEqual(st.route_status, config.ROUTE_FAILOVER)

        # Route change must NOT cause abnormal sensor behaviour fault (FAULT 7)
        self.assertNotIn(config.FAULT_ABNORMAL_SENSOR, st.faults)

        # Extreme spike is sent on sensor readings
        self.feed(1, 17, temp=42.0, hum=50.0, light=1800, via=2, hops=1, rx_ts=now + 34)
        # Processor identified rate anomaly on temperature
        processed_msgs = self.p.topics(config.t_processed(1))
        latest_proc = processed_msgs[-1]
        self.assertTrue(latest_proc["anomaly"])


if __name__ == "__main__":
    unittest.main()

