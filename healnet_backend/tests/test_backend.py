"""Offline tests for the HEALNET backend logic (no broker, no hardware needed).

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.DB_PATH = os.path.join(tempfile.mkdtemp(), "t.db")
config.STARTUP_GRACE_S = 0
config.ML_MIN_SAMPLES = 10
config.ML_IFOREST_MIN_SAMPLES = 10_000          # keep unit tests deterministic (statistics only)

import processor as P  # noqa: E402
import protocol  # noqa: E402


class Capture(P.Processor):
    def __init__(self):
        config.DB_PATH = os.path.join(tempfile.mkdtemp(), "t.db")
        super().__init__()
        self.out = []

    def pub(self, topic, payload, qos=0, retain=False):
        self.out.append((topic, payload))

    def topics(self, t):
        return [p for (x, p) in self.out if x == t]


def pkt(node, seq, **kw):
    d = {"node": node, "seq": seq, "temp": 26.0, "hum": 55.0, "light": 1800, "valid": True,
         "uptime_ms": seq * 2000, "rx_ts": time.time()}
    d.update(kw)
    return d


class Tests(unittest.TestCase):
    def setUp(self):
        self.p = Capture()

    def feed(self, node, seq, **kw):
        self.p._on_data(config.t_data(node), pkt(node, seq, **kw))

    def test_duplicate_is_dropped(self):
        self.feed(1, 1); self.feed(1, 1)
        self.assertEqual(self.p.nodes[1].dup, 1)
        self.assertEqual(self.p.nodes[1].packets, 1)

    def test_gap_counts_lost_packets(self):
        self.feed(1, 1); self.feed(1, 6)
        self.assertEqual(self.p.nodes[1].lost, 4)
        self.assertTrue(self.p.topics(config.T_FAULT))        # >= GAP_EVENT_MIN -> PACKET_LOSS event

    def test_out_of_order_late_arrival(self):
        self.feed(1, 1); self.feed(1, 3); self.feed(1, 2)
        st = self.p.nodes[1]
        self.assertEqual((st.ooo, st.dup), (1, 0))
        self.assertEqual(st.last_seq, 3)

    def test_reboot_detected_by_uptime_and_seq(self):
        for s in range(1, 40):
            self.feed(1, s)
        self.feed(1, 0, uptime_ms=500)
        self.assertEqual(self.p.nodes[1].reboots, 1)
        self.feed(1, 1, uptime_ms=2500)                         # seq 1 is NOT a duplicate after a reboot
        self.assertEqual(self.p.nodes[1].dup, 0)

    def test_rejects_node_id_mismatch_and_bad_seq(self):
        self.p._on_data(config.t_data(1), pkt(2, 1))
        self.p._on_data(config.t_data(1), pkt(1, "x"))
        self.assertEqual(self.p.counters["rejected"], 2)

    def test_dht_failure_opens_and_clears_fault1(self):
        for s in range(1, 4):
            self.feed(1, s)
        for s in range(4, 8):
            self.feed(1, s, temp=None, hum=None, valid=False)
        self.assertIn(1, self.p.nodes[1].faults)
        self.assertEqual(self.p.nodes[1].sensor_state["dht"], "FAILED")
        for s in range(8, 12):
            self.feed(1, s)
        self.assertNotIn(1, self.p.nodes[1].faults)

    def test_ldr_pinned_at_rail_becomes_suspect(self):
        for s in range(1, config.LDR_RAIL_STREAK + 3):
            self.feed(2, s, light=4095)
        self.assertEqual(self.p.nodes[2].sensor_state["ldr"], "SUSPECT")
        self.assertIn(1, self.p.nodes[2].faults)

    def test_node_timeout_goes_offline_with_fault2_and_recovers(self):
        self.feed(1, 1)
        self.p.coordinator_online = True
        self.p.nodes[1].last_rx_ts = time.time() - 12
        self.p.watchdog_tick()
        self.assertEqual(self.p.nodes[1].status, P.OFFLINE)
        self.assertIn(2, self.p.nodes[1].faults)
        # 2 consecutive valid direct packets required to confirm recovery (RECOVERY_VALIDATION_PACKETS=2)
        self.feed(1, 2)
        self.feed(1, 3)
        self.assertEqual(self.p.nodes[1].status, P.ONLINE)
        self.assertNotIn(2, self.p.nodes[1].faults)

    def test_coordinator_down_is_fault5_not_node_fault(self):
        self.feed(1, 1); self.feed(2, 1)
        self.p._on_coordinator({"status": "offline", "last_line_age_s": 9, "port": "X"})
        for n in (1, 2):
            self.p.nodes[n].last_rx_ts = time.time() - 30
        self.p.watchdog_tick()
        self.assertIn(5, self.p._system_state().faults)
        for n in (1, 2):
            self.assertEqual(self.p.nodes[n].status, P.UNKNOWN)
            self.assertNotIn(2, self.p.nodes[n].faults)
        self.p._on_coordinator({"status": "online"})
        self.assertNotIn(5, self.p._system_state().faults)

    def test_delayed_backfill_does_not_change_live_state(self):
        self.feed(1, 5)
        before = self.p.nodes[1].last_rx_ts
        self.feed(1, 4, rx_ts=time.time() - 60)
        self.assertEqual(self.p.nodes[1].last_rx_ts, before)
        self.assertEqual(self.p.nodes[1].lost, 0)
        self.assertEqual(self.p.nodes[1].delayed, 1)

    def test_route_change_published(self):
        self.feed(1, 1)
        self.feed(1, 2, via=2, hops=1)
        r = self.p.topics(config.T_ROUTING)
        self.assertIn(r[-1]["state"], ("REROUTED", "FAILOVER"))
        self.assertEqual(r[-1]["path"], [1, 2, "C"])

    def test_nan_is_sanitised_by_bridge_protocol(self):
        msgs = protocol.translate({"type": "data", "node": 1, "seq": 1, "temp": float("nan")}, time.time(), 1)
        self.assertIsNone(msgs[0][1]["temp"])

    def test_serial_noise_is_ignored(self):
        self.assertIsNone(protocol.parse_line("ESP-NOW init ok"))
        self.assertIsNone(protocol.parse_line('{"broken": '))
        self.assertEqual(protocol.parse_line('{"type":"hb"}')["type"], "hb")


class AnomalyTests(unittest.TestCase):
    def warm(self, eng, n=40):
        t = time.time() - 200
        for i in range(n):
            for node in (1, 2):
                eng.process(node, t + i * 2, 26 + (i % 3) * 0.1, 55, 1800 + (i % 5) * 5)
        return t + n * 2

    def test_only_one_node_deviating_is_localized(self):
        from anomaly import AnomalyEngine
        eng = AnomalyEngine((1, 2))
        t = self.warm(eng)
        r = eng.process(1, t, 41.0, 55, 1800)
        eng.process(2, t + 0.5, 26, 55, 1800)
        self.assertTrue(r.anomalous)
        self.assertEqual(eng.classify(1, r.findings[0], t), "localized_sensor_suspect")

    def test_both_nodes_moving_together_is_environmental(self):
        from anomaly import AnomalyEngine
        eng = AnomalyEngine((1, 2))
        t = self.warm(eng)
        r1 = eng.process(1, t, 26, 55, 600)
        eng.process(2, t + 0.7, 26, 55, 650)
        self.assertTrue(r1.anomalous)
        self.assertEqual(eng.classify(1, r1.findings[0], t), "environmental_change")

    def test_normal_data_is_not_flagged(self):
        from anomaly import AnomalyEngine
        eng = AnomalyEngine((1, 2))
        t = self.warm(eng)
        self.assertFalse(eng.process(1, t, 26.1, 55, 1810).anomalous)


def coord_pkt(node_id, sequence, **kw):
    """Generates message in current coordinator format."""
    d = {
        "type": "sensor",
        "node_id": node_id,
        "temperature": 29.5,
        "humidity": 45.4,
        "light": 834,
        "sequence": sequence,
        "uptime_ms": sequence * 2000,
    }
    d.update(kw)
    return d


class CoordinatorFormatTests(unittest.TestCase):
    def setUp(self):
        self.p = Capture()

    def feed_coord(self, node_id, sequence, **kw):
        payload = coord_pkt(node_id, sequence, **kw)
        self.p._on_data(config.t_data(node_id), payload)

    def test_node1_and_node2_messages_processed_and_stored(self):
        # Node 1 message
        self.feed_coord(1, 317, temperature=29.5, humidity=45.4, light=834, uptime_ms=155476)
        # Node 2 message
        self.feed_coord(2, 105, temperature=24.2, humidity=60.1, light=1200, uptime_ms=155500)

        # Check in-memory node state
        st1 = self.p.nodes[1]
        st2 = self.p.nodes[2]
        self.assertEqual(st1.status, P.ONLINE)
        self.assertEqual(st1.last_seq, 317)
        self.assertEqual(st1.last_uptime, 155476)
        self.assertEqual(st1.packets, 1)

        self.assertEqual(st2.status, P.ONLINE)
        self.assertEqual(st2.last_seq, 105)
        self.assertEqual(st2.last_uptime, 155500)
        self.assertEqual(st2.packets, 1)

        # Verify SQLite storage
        rows1 = self.p.db.recent_readings(1, 10)
        self.assertEqual(len(rows1), 1)
        self.assertEqual(rows1[0]["node_id"], 1)
        self.assertEqual(rows1[0]["seq"], 317)
        self.assertEqual(rows1[0]["temp"], 29.5)
        self.assertEqual(rows1[0]["hum"], 45.4)
        self.assertEqual(rows1[0]["light"], 834)
        self.assertEqual(rows1[0]["uptime_ms"], 155476)

        rows2 = self.p.db.recent_readings(2, 10)
        self.assertEqual(len(rows2), 1)
        self.assertEqual(rows2[0]["node_id"], 2)
        self.assertEqual(rows2[0]["seq"], 105)
        self.assertEqual(rows2[0]["temp"], 24.2)
        self.assertEqual(rows2[0]["hum"], 60.1)
        self.assertEqual(rows2[0]["light"], 1200)

        # Verify processed publication on MQTT
        p1 = self.p.topics(config.t_processed(1))
        self.assertTrue(len(p1) >= 1)
        self.assertEqual(p1[-1]["node"], 1)
        self.assertEqual(p1[-1]["seq"], 317)
        self.assertEqual(p1[-1]["temp"], 29.5)
        self.assertEqual(p1[-1]["hum"], 45.4)
        self.assertEqual(p1[-1]["light"], 834)

        p2 = self.p.topics(config.t_processed(2))
        self.assertTrue(len(p2) >= 1)
        self.assertEqual(p2[-1]["node"], 2)
        self.assertEqual(p2[-1]["seq"], 105)
        self.assertEqual(p2[-1]["temp"], 24.2)
        self.assertEqual(p2[-1]["hum"], 60.1)
        self.assertEqual(p2[-1]["light"], 1200)

    def test_invalid_readings(self):
        # 1. Invalid sequence
        self.p._on_data(config.t_data(1), {"type": "sensor", "node_id": 1, "sequence": "not-an-int"})
        self.assertEqual(self.p.counters["rejected"], 1)

        # 2. Node mismatch
        self.p._on_data(config.t_data(1), coord_pkt(2, 100))
        self.assertEqual(self.p.counters["rejected"], 2)

        # 3. Non-numeric / NaN reading sanitized without crashing
        self.feed_coord(1, 1, temperature=float("nan"))
        rows = self.p.db.recent_readings(1, 1)
        self.assertIsNone(rows[0]["temp"])
        self.assertEqual(rows[0]["dht_ok"], 0)

    def test_duplicate_sequence_numbers(self):
        self.feed_coord(1, 317)
        self.feed_coord(1, 317)
        st = self.p.nodes[1]
        self.assertEqual(st.packets, 1)
        self.assertEqual(st.dup, 1)

    def test_sequence_gaps(self):
        self.feed_coord(1, 317)
        self.feed_coord(1, 325)  # gap of 7 packets
        st = self.p.nodes[1]
        self.assertEqual(st.lost, 7)
        faults = self.p.topics(config.T_FAULT)
        loss_events = [f for f in faults if f.get("event") == "PACKET_LOSS"]
        self.assertEqual(len(loss_events), 1)
        self.assertEqual(loss_events[0]["missing"], 7)
        self.assertEqual(loss_events[0]["from_seq"], 318)
        self.assertEqual(loss_events[0]["to_seq"], 324)

    def test_sensor_failures(self):
        # Good packets
        for s in range(1, 4):
            self.feed_coord(1, s)
        # 3 bad packets -> DHT failure (FAULT 1)
        for s in range(4, 7):
            self.feed_coord(1, s, temperature=None, humidity=None)
        st = self.p.nodes[1]
        self.assertIn(1, st.faults)
        self.assertEqual(st.sensor_state["dht"], "FAILED")

        # LDR rail test on node 2
        for s in range(1, config.LDR_RAIL_STREAK + 3):
            self.feed_coord(2, s, light=4095)
        st2 = self.p.nodes[2]
        self.assertEqual(st2.sensor_state["ldr"], "SUSPECT")
        self.assertIn(1, st2.faults)

    def test_communication_timeout(self):
        self.feed_coord(1, 1)
        self.p.coordinator_online = True
        self.p.nodes[1].last_rx_ts = time.time() - 15
        self.p.watchdog_tick()
        st = self.p.nodes[1]
        self.assertEqual(st.status, P.OFFLINE)
        self.assertIn(2, st.faults)
        self.assertIn("not determined by passive monitoring", st.faults[2]["detail"])

    def test_coordinator_disconnection_and_node_isolation(self):
        self.feed_coord(1, 1)
        self.feed_coord(2, 1)
        # Coordinator goes offline
        self.p._on_coordinator({"type": "coordinator", "status": "offline", "last_line_age_s": 8.5, "port": "COM3"})
        self.assertIn(5, self.p._system_state().faults)

        # Trigger watchdog tick after timeout
        for n in (1, 2):
            self.p.nodes[n].last_rx_ts = time.time() - 25
        self.p.watchdog_tick()

        # Both nodes must be UNKNOWN, NOT OFFLINE, and FAULT 2 must NOT be raised
        for n in (1, 2):
            self.assertEqual(self.p.nodes[n].status, P.UNKNOWN)
            self.assertNotIn(2, self.p.nodes[n].faults)

        # Coordinator recovers
        self.p._on_coordinator({"type": "coordinator", "status": "online"})
        self.assertNotIn(5, self.p._system_state().faults)

    def test_recovery_when_sensor_packets_resume(self):
        # 1. Recovery from node communication timeout
        self.feed_coord(1, 1)
        self.p.coordinator_online = True
        self.p.nodes[1].last_rx_ts = time.time() - 15
        self.p.watchdog_tick()
        self.assertEqual(self.p.nodes[1].status, P.OFFLINE)
        self.assertIn(2, self.p.nodes[1].faults)

        # Resuming packets clears FAULT 2 and marks node ONLINE after 2 consecutive valid packets
        self.feed_coord(1, 2)
        self.feed_coord(1, 3)
        self.assertEqual(self.p.nodes[1].status, P.ONLINE)
        self.assertNotIn(2, self.p.nodes[1].faults)

        # 2. Recovery from sensor failure
        for s in range(4, 8):
            self.feed_coord(1, s, temperature=None, humidity=None)
        self.assertIn(1, self.p.nodes[1].faults)
        # 3 good packets clear FAULT 1
        for s in range(8, 12):
            self.feed_coord(1, s, temperature=28.0, humidity=50.0, light=800)
        self.assertNotIn(1, self.p.nodes[1].faults)
        self.assertEqual(self.p.nodes[1].sensor_state["dht"], "OK")

    def test_protocol_translation_current_coordinator_format(self):
        line = '{"type":"sensor","node_id":1,"temperature":29.5,"humidity":45.4,"light":834,"sequence":317,"uptime_ms":155476}'
        parsed = protocol.parse_line(line)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["type"], "sensor")
        self.assertEqual(parsed["node_id"], 1)

        pubs = protocol.translate(parsed, rx_ts=1700000000.123, bridge_seq=1)
        self.assertEqual(len(pubs), 1)
        topic, envelope, qos, retain = pubs[0]
        self.assertEqual(topic, "healnet/node1/data")
        self.assertEqual(qos, 1)
        self.assertFalse(retain)
        self.assertEqual(envelope["node"], 1)
        self.assertEqual(envelope["seq"], 317)
        self.assertEqual(envelope["temp"], 29.5)
        self.assertEqual(envelope["hum"], 45.4)
        self.assertEqual(envelope["light"], 834)
        self.assertEqual(envelope["uptime_ms"], 155476)
        self.assertEqual(envelope["rx_ts"], 1700000000.123)
        self.assertEqual(envelope["bridge_seq"], 1)
        self.assertEqual(envelope["source"], "serial")


if __name__ == "__main__":
    unittest.main()
