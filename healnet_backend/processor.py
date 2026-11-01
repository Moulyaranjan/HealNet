"""
HEALNET backend processor.

Subscribes to raw sensor packets on MQTT, then:
  validate -> de-duplicate -> sequence analysis -> sensor-fault checks ->
  anomaly detection (ML) -> store in SQLite -> publish enriched data, node
  status, faults, anomalies, routing changes and a network summary.

Deterministic fault detection (timeouts, coordinator liveness, sensor validity)
decides about FAULTS. The ML result only ever produces "abnormal sensor
behaviour" (FAULT 7); it never marks a node as failed.

Run:  python processor.py
"""
import json
import logging
import signal
import threading
import time
from collections import deque

import config
import protocol
from anomaly import AnomalyEngine
from edge_metrics import get_metrics
from mqtt_utils import connect_async, make_client, publish_json
from storage import Storage

try:
    from ml_pipeline import ModelManager
    _HAVE_ML_MODEL = True
except ImportError:
    _HAVE_ML_MODEL = False
    ModelManager = None

log = logging.getLogger("healnet.processor")

UNKNOWN, ONLINE, STALE, OFFLINE = "UNKNOWN", "ONLINE", "STALE", "OFFLINE"
# RECOVERING / FAULT states are reserved for the recovery phase.


class NodeState:
    def __init__(self, node_id):
        self.node_id = node_id
        self.status = UNKNOWN
        self.status_since = time.time()
        self.last_rx_ts = 0.0            # bridge receive time of the newest live packet
        self.last_seq = None
        self.last_uptime = None
        self.last_boot = None
        self.recent = deque(maxlen=config.DEDUP_WINDOW)
        self.recent_set = set()
        self.packets = self.dup = self.ooo = self.lost = self.reboots = self.rejected = self.delayed = 0
        self.last_payload = {}
        self.rssi = None
        self.route = (None, 0)           # (via, hops)
        # sensor-health counters
        self.dht_bad = self.dht_good = 0
        self.ldr_rail = self.ldr_good = 0
        # anomaly persistence
        self.recent_flags = deque(maxlen=10)
        self.local_hits = deque()        # wall-clock times of "localized_sensor_suspect" samples
        self.clean_run = 0
        self.faults = {}                 # fault_code -> {"id":db_id,"detail":str}
        self.sensor_state = {"dht": "OK", "ldr": "OK"}

    def remember(self, seq):
        if len(self.recent) == self.recent.maxlen:
            self.recent_set.discard(self.recent[0])
        self.recent.append(seq)
        self.recent_set.add(seq)

    def reset_sequence(self):
        self.recent.clear()
        self.recent_set.clear()
        self.last_seq = None


class Processor:
    def __init__(self):
        self.db = Storage(config.DB_PATH)
        self.engine = AnomalyEngine(config.NODE_IDS)
        self.nodes = {n: NodeState(n) for n in config.NODE_IDS}
        self.lock = threading.RLock()
        self.started = time.time()
        self.stop = threading.Event()
        self.coordinator_online = None       # None = not heard yet
        self.bridge_online = None
        self._sys = NodeState(0)               # node_id 0 = system-wide faults (coordinator, backend)
        self._coord_detail = ""
        self.coord_back_ts = 0.0
        self.mqtt_ok = False
        self.counters = {"messages": 0, "rejected": 0, "published": 0}
        self.pending = deque()               # anomalies waiting for cross-node classification
        self.metrics = get_metrics()

        self.ml_model = ModelManager() if _HAVE_ML_MODEL else None
        if self.ml_model:
            try:
                self.ml_model.load()
            except Exception as exc:
                log.warning("Could not pre-load ML model: %s", exc)

        self.client = make_client("healnet-processor", config.T_BACKEND, {"status": "offline"}, persistent=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.db.insert_system_event("backend_startup", "HEALNET processor initialized", "INFO")
        self._warm_start()

    # =====================================================================
    # start-up
    # =====================================================================
    def _warm_start(self):
        """Reload recent history so ML windows and sequence tracking survive a restart."""
        self.db.close_stale_faults(time.time())
        for n, st in self.nodes.items():
            rows = self.db.recent_readings(n, config.ML_WINDOW)
            if rows:
                self.engine.seed(n, rows)
                last = rows[-1]
                st.last_seq, st.last_rx_ts = last["seq"], last["ts"]
                st.last_uptime, st.last_boot = last["uptime_ms"], last["boot"]
                for r in rows[-config.DEDUP_WINDOW:]:
                    if r["seq"] is not None:
                        st.remember(r["seq"])
                log.info("node%d: restored %d readings (last seq %s)", n, len(rows), st.last_seq)

    # =====================================================================
    # MQTT
    # =====================================================================
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connect refused: %s", reason_code)
            self.db.insert_system_event("mqtt_refused", f"Connect refused: {reason_code}", "ERROR")
            return
        self.mqtt_ok = True
        log.info("MQTT connected (session_present=%s)", flags.session_present)
        self.db.insert_system_event("mqtt_connect", f"Connected to {config.MQTT_HOST}:{config.MQTT_PORT}", "INFO")
        if config.FAULT_BACKEND in self._sys.faults:
            self._clear_fault(self._sys, config.FAULT_BACKEND, "MQTT broker reconnected")
        for n in config.NODE_IDS:
            client.subscribe(config.t_data(n), qos=1)
        client.subscribe(config.T_COORDINATOR, qos=1)
        client.subscribe(config.T_BRIDGE, qos=1)
        client.subscribe(config.T_COORD_EVENT, qos=1)
        publish_json(client, config.T_BACKEND, {"status": "online", "since": time.time()}, qos=1, retain=True)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.mqtt_ok = False
        log.warning("MQTT disconnected (%s) - paho will keep reconnecting; broker queues QoS1 for us", reason_code)
        self.db.insert_system_event("mqtt_disconnect", f"Disconnected from MQTT broker ({reason_code})", "WARNING")
        self._open_fault(self._sys, config.FAULT_BACKEND, f"MQTT disconnected ({reason_code})")

    def _on_message(self, client, userdata, msg):
        with self.metrics.track("mqtt_processing"):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("payload is not an object")
            except Exception as exc:
                self.counters["rejected"] += 1
                log.warning("bad JSON on %s: %s", msg.topic, exc)
                return
            self.counters["messages"] += 1
            self.metrics.inc_messages(1)
            try:
                with self.lock:
                    if msg.topic == config.T_COORDINATOR:
                        self._on_coordinator(payload)
                    elif msg.topic == config.T_BRIDGE:
                        self._on_bridge(payload)
                    elif msg.topic == config.T_COORD_EVENT:
                        self._on_coordinator_event(payload)
                    else:
                        self._on_data(msg.topic, payload)
            except Exception:
                log.exception("processing error on %s", msg.topic)

    def _on_coordinator_event(self, p):
        log.info("coordinator event: %s", p)
        ev_type = str(p.get("event", "generic_event")).lower()
        node_id = p.get("node_id") or p.get("node")
        msg = str(p.get("message") or p.get("detail", ""))
        now = time.time()

        if "recovery" in ev_type or "recovery" in msg.lower():
            self.db.insert_recovery_event(
                now, node_id, method="manual_button", trigger_type="coordinator_button",
                result="SUCCESS", detail=f"{ev_type}: {msg}")
            self.db.insert_system_event(
                "coordinator_recovery", f"Recovery on node {node_id}: {msg}", "INFO")
            self.pub(config.T_RECOVERY, {
                "ts": round(now, 3), "node": node_id, "event": ev_type,
                "method": "manual_button", "result": "SUCCESS", "detail": msg
            }, qos=1)
        elif "fail" in ev_type:
            self.db.insert_system_event(
                "simulated_failure", f"Failure simulation on node {node_id}: {msg}", "WARNING")
            self.pub(config.T_FAULT, {
                "ts": round(now, 3), "event": "SIMULATED_FAILURE", "node": node_id,
                "severity": "WARNING", "detail": msg, "state": "EVENT"
            }, qos=1)
        else:
            self.db.insert_system_event("coordinator_event", f"{ev_type}: {msg}", "INFO")

    def pub(self, topic, payload, qos=0, retain=False):
        if publish_json(self.client, topic, payload, qos, retain):
            self.counters["published"] += 1

    # =====================================================================
    # data path
    # =====================================================================
    def _validate(self, topic, p):
        """Return (clean_dict, None) or (None, reason)."""
        p = protocol.adapt_sensor_payload(p)
        try:
            topic_node = int(topic.split("/")[1].replace("node", ""))
        except (IndexError, ValueError):
            return None, "bad topic"
        node = p.get("node", topic_node)
        if node != topic_node or node not in self.nodes:
            return None, f"node id mismatch/unknown (topic {topic_node}, payload {node})"
        seq = p.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            return None, "missing/invalid seq"
        clean = {"node": node, "seq": seq}
        clean["rx_ts"] = float(p["rx_ts"]) if isinstance(p.get("rx_ts"), (int, float)) else time.time()
        for key in ("uptime_ms", "boot", "rssi", "via", "hops", "bridge_seq"):
            if isinstance(p.get(key), (int, float)) and not isinstance(p.get(key), bool):
                clean[key] = int(p[key])
        clean["hops"] = clean.get("hops", 0)
        clean["valid"] = bool(p.get("valid", True))
        # sensor values: keep only finite numbers
        for key in ("temp", "hum", "light"):
            v = p.get(key)
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")
            clean[key] = float(v) if ok else None
        clean["light"] = None if clean["light"] is None else int(clean["light"])
        # Per-sensor health. If the firmware only sends one "valid" flag it can only be the DHT11
        # (an analog LDR cannot self-report), so valid=false is attributed to the DHT11; the LDR is
        # judged from its value (missing / pinned at an ADC rail).
        dht_values = clean["temp"] is not None and clean["hum"] is not None
        clean["dht_ok"] = bool(p["dht_ok"]) if "dht_ok" in p else (clean["valid"] and dht_values)
        clean["ldr_ok"] = bool(p["ldr_ok"]) if "ldr_ok" in p else (clean["light"] is not None)
        return clean, None

    def _on_data(self, topic, p):
        clean, reason = self._validate(topic, p)
        if clean is None:
            self.counters["rejected"] += 1
            log.warning("rejected packet on %s: %s", topic, reason)
            return
        st = self.nodes[clean["node"]]
        now = time.time()
        flags = []

        delayed = (now - clean["rx_ts"]) > config.DELAYED_PACKET_S
        if delayed:
            flags.append("delayed")
            st.delayed += 1

        # ---- duplicate / reboot / ordering -------------------------------
        boot = clean.get("boot")
        uptime = clean.get("uptime_ms")
        seq = clean["seq"]
        rebooted = False
        if not delayed:
            if boot is not None and st.last_boot is not None and boot != st.last_boot:
                rebooted = True
            elif (uptime is not None and st.last_uptime is not None
                  and uptime + (config.OOO_WINDOW + 1) * config.SENSOR_INTERVAL_S * 1000 < st.last_uptime):
                rebooted = True
            elif st.last_seq is not None and seq < st.last_seq - config.OOO_WINDOW and seq not in st.recent_set:
                rebooted = True
        if rebooted:
            st.reboots += 1
            log.info("node%d REBOOT detected (seq %s -> %s)", st.node_id, st.last_seq, seq)
            st.reset_sequence()
            flags.append("reboot")

        if seq in st.recent_set:
            st.dup += 1
            log.debug("node%d duplicate seq %d dropped", st.node_id, seq)
            return                                     # QoS1 redelivery / forwarded copy / replay

        if st.last_seq is not None and not rebooted and not delayed:
            if seq > st.last_seq + 1:
                gap = seq - st.last_seq - 1
                st.lost += gap
                flags.append("gap")
                if gap >= config.GAP_EVENT_MIN and not delayed:
                    self.pub(config.T_FAULT, {
                        "ts": round(now, 3), "event": "PACKET_LOSS", "node": st.node_id, "missing": gap,
                        "from_seq": st.last_seq + 1, "to_seq": seq - 1, "state": "EVENT", "severity": "LOW"}, qos=1)
            elif seq < st.last_seq:
                st.ooo += 1
                flags.append("out_of_order")

        st.remember(seq)
        st.packets += 1

        # ---- late/backfilled packets: store history only ------------------
        if delayed:
            if st.last_seq is None or seq > st.last_seq:
                st.last_seq = seq                      # backfill fills the gap; do not count it as loss
            clean["flags"] = flags
            self.db.insert_reading(self._row(clean))
            return

        if st.last_seq is None or seq > st.last_seq or rebooted:
            st.last_seq = seq
        st.last_rx_ts = max(st.last_rx_ts, clean["rx_ts"])
        st.last_uptime, st.last_boot = uptime if uptime is not None else st.last_uptime, \
            boot if boot is not None else st.last_boot
        st.rssi = clean.get("rssi", st.rssi)

        self._set_status(st, ONLINE, "packet received")
        self._check_route(st, clean)
        with self.metrics.track("fault_detection"):
            self._check_sensors(st, clean)

        # ---- anomaly detection (only on trustworthy readings) -------------
        anomaly = None
        ml_anom, ml_score = False, 0.0
        usable = (clean["valid"] and clean["dht_ok"] and clean["ldr_ok"]
                  and None not in (clean["temp"], clean["hum"], clean["light"])
                  and self._in_range(clean))
        if usable:
            with self.metrics.track("ml_inference"):
                anomaly = self.engine.process(st.node_id, clean["rx_ts"], clean["temp"], clean["hum"], clean["light"])
                if self.ml_model:
                    try:
                        ml_anom, ml_score = self.ml_model.predict(
                            clean["temp"], clean["hum"], clean["light"], node_id=st.node_id
                        )
                        if ml_anom:
                            self.metrics.inc_anomalies(1)
                    except Exception as exc:
                        log.debug("ML model inference error: %s", exc)

            self._handle_anomaly(st, clean, anomaly)
            clean["anomaly"] = anomaly.anomalous
            clean["anomaly_score"] = anomaly.score

        clean["ml_anomaly"] = ml_anom
        clean["ml_score"] = ml_score
        clean["flags"] = flags
        with self.metrics.track("db_write"):
            self.db.insert_reading(self._row(clean))

        out = {
            "node": st.node_id, "seq": seq, "ts": clean["rx_ts"],
            "temp": clean["temp"], "hum": clean["hum"], "light": clean["light"],
            "valid": clean["valid"], "dht_ok": clean["dht_ok"], "ldr_ok": clean["ldr_ok"],
            "rssi": st.rssi, "via": clean.get("via"), "hops": clean["hops"],
            "flags": flags, "anomaly": bool(anomaly and anomaly.anomalous),
            "anomaly_score": anomaly.score if anomaly else None,
            "anomaly_class": "pending" if (anomaly and anomaly.anomalous) else None,
            "ml_anomaly": ml_anom,
            "ml_score": ml_score,
            "status": st.status,
        }
        self.pub(config.t_processed(st.node_id), out, qos=0)
        st.last_payload = out

    @staticmethod
    def _row(c):
        return c

    @staticmethod
    def _in_range(c):
        t0, t1 = config.TEMP_RANGE
        h0, h1 = config.HUM_RANGE
        l0, l1 = config.LIGHT_RANGE
        return t0 <= c["temp"] <= t1 and h0 <= c["hum"] <= h1 and l0 <= c["light"] <= l1

    # =====================================================================
    # FAULT 1 - sensor failure (deterministic)
    # =====================================================================
    def _check_sensors(self, st, c):
        # DHT11: invalid flag, missing value, or physically impossible value
        dht_bad = (not c["dht_ok"]) or c["temp"] is None or c["hum"] is None
        if not dht_bad:
            dht_bad = not (config.TEMP_RANGE[0] <= c["temp"] <= config.TEMP_RANGE[1]
                           and config.HUM_RANGE[0] <= c["hum"] <= config.HUM_RANGE[1])
        if dht_bad:
            st.dht_bad += 1
            st.dht_good = 0
        else:
            st.dht_good += 1
            st.dht_bad = 0

        # LDR: flag from node, missing value, or pinned at an ADC rail for a long time
        light = c["light"]
        rail = light is not None and (light <= config.LDR_RAIL_LOW or light >= config.LDR_RAIL_HIGH)
        ldr_bad = (not c["ldr_ok"]) or light is None
        if ldr_bad or rail:
            st.ldr_rail += 1
            st.ldr_good = 0
        else:
            st.ldr_good += 1
            st.ldr_rail = 0

        dht_fault = st.dht_bad >= config.SENSOR_INVALID_STREAK
        ldr_hard = ldr_bad and st.ldr_rail >= config.SENSOR_INVALID_STREAK
        ldr_suspect = st.ldr_rail >= config.LDR_RAIL_STREAK
        st.sensor_state["dht"] = "FAILED" if dht_fault else "OK"
        st.sensor_state["ldr"] = "FAILED" if ldr_hard else ("SUSPECT" if ldr_suspect else "OK")

        if dht_fault or ldr_hard or ldr_suspect:
            parts = []
            if dht_fault:
                parts.append(f"DHT11 invalid/out-of-range for {st.dht_bad} consecutive packets")
            if ldr_hard:
                parts.append(f"LDR reports invalid for {st.ldr_rail} consecutive packets")
            elif ldr_suspect:
                parts.append(f"LDR pinned at ADC rail ({light}) for {st.ldr_rail} packets - disconnected or saturated?")
            self._open_fault(st, 1, "; ".join(parts))
        elif 1 in st.faults and st.dht_good >= config.SENSOR_RECOVER_STREAK \
                and st.ldr_good >= config.SENSOR_RECOVER_STREAK:
            self._clear_fault(st, 1, "sensors reporting valid data again")

    # =====================================================================
    # FAULT 7 - abnormal sensor behaviour (ML + statistics)
    # =====================================================================
    def _handle_anomaly(self, st, c, result):
        st.recent_flags.append(result.anomalous)
        if result.anomalous:
            self.metrics.inc_anomalies(1)
            st.clean_run = 0
            # Wait one sensor interval (+margin) before classifying, so the other node's
            # sample for the same moment has arrived. See release_pending().
            self.pending.append((time.time() + config.SENSOR_INTERVAL_S + 0.8, st.node_id, c["rx_ts"],
                                 c["seq"], result.findings))
        else:
            st.clean_run += 1
            if 7 in st.faults and st.clean_run >= config.ANOMALY_CLEAR_COUNT:
                self._clear_fault(st, 7, "readings back within normal behaviour")

    def release_pending(self, now):
        """Classify and publish anomalies whose cross-node waiting time is over."""
        while self.pending and self.pending[0][0] <= now:
            _, node, ts, seq, findings = self.pending.popleft()
            st = self.nodes[node]
            local = False
            for f in findings:
                f.classification = self.engine.classify(node, f, ts)
                local = local or f.classification == "localized_sensor_suspect"
                self.db.insert_anomaly(ts, node, f.feature, f.method, round(f.score, 3), f.classification, f.detail)
                self.pub(config.T_ANOMALY, {
                    "ts": round(ts, 3), "node": node, "seq": seq, "feature": f.feature, "method": f.method,
                    "score": round(f.score, 3), "classification": f.classification, "detail": f.detail,
                    "note": "sensor-behaviour anomaly only - this is NOT a node/hardware failure"}, qos=1)
            if local:
                st.local_hits.append(now)
            while st.local_hits and now - st.local_hits[0] > 20:
                st.local_hits.popleft()
            if local and len(st.local_hits) >= config.ANOMALY_PERSIST_COUNT:
                self._open_fault(st, 7, f"sustained abnormal readings on this node only: {findings[0].detail}")

    # =====================================================================
    # routing (future multi-hop; passes through whatever the coordinator reports)
    # =====================================================================
    def _check_route(self, st, c):
        route = (c.get("via"), c["hops"])
        if route != st.route:
            old, st.route = st.route, route
            state = "DIRECT" if route[0] is None and route[1] == 0 else "REROUTED"
            path = [st.node_id] + ([route[0]] if route[0] is not None else []) + ["C"]
            self.db.insert_route(c["rx_ts"], st.node_id, route[0], route[1], state)
            self.pub(config.T_ROUTING, {
                "ts": round(c["rx_ts"], 3), "node": st.node_id, "state": state, "via": route[0],
                "hops": route[1], "path": path, "previous": {"via": old[0], "hops": old[1]}}, qos=1)
            log.info("node%d route %s via=%s hops=%s", st.node_id, state, route[0], route[1])

    # =====================================================================
    # coordinator / bridge liveness  ->  FAULT 5
    # =====================================================================
    def _on_coordinator(self, p):
        status_val = p.get("status", p.get("state"))
        if status_val is not None:
            online = str(status_val).strip().lower() in ("online", "ok", "up", "connected", "true", "1")
        elif "online" in p:
            online = bool(p["online"])
        elif p.get("type") in ("hb", "heartbeat") or "uptime_ms" in p:
            online = True
        else:
            online = False

        if online != self.coordinator_online:
            log.info("coordinator %s", "ONLINE" if online else "OFFLINE")
        self.coordinator_online = online
        if not online:
            age = p.get("last_line_age_s")
            port = p.get("port")
            detail = p.get("detail") or p.get("reason")
            if detail:
                self._coord_detail = f"coordinator offline: {detail}"
            elif age is not None:
                self._coord_detail = f"no serial data from coordinator (last line {age} s ago, port {port})"
            else:
                self._coord_detail = "coordinator reported offline"
        self._update_coordinator_fault()

    def _on_bridge(self, p):
        status_val = p.get("status", p.get("state"))
        if status_val is not None:
            online = str(status_val).strip().lower() in ("online", "ok", "up", "connected", "true", "1")
        elif "online" in p:
            online = bool(p["online"])
        else:
            online = False

        if online != self.bridge_online:
            log.info("bridge %s", "ONLINE" if online else "OFFLINE")
        self.bridge_online = online
        if not self.bridge_online:
            self._coord_detail = "serial bridge offline - no data path from the coordinator"
        self._update_coordinator_fault()

    def _update_coordinator_fault(self):
        """FAULT 5 is active while the bridge or coordinator is reported down."""
        down = self.bridge_online is False or self.coordinator_online is False
        sysst = self._system_state()
        if down:
            self._open_fault(sysst, 5, self._coord_detail)
        elif self.coordinator_online and 5 in sysst.faults:
            self._clear_fault(sysst, 5, "coordinator sending data again")
            self.coord_back_ts = time.time()

    def _system_state(self):
        return self._sys

    # =====================================================================
    # fault bookkeeping
    # =====================================================================
    def _open_fault(self, st, code, detail):
        if code in st.faults:
            st.faults[code]["detail"] = detail
            return
        name, severity = config.FAULTS[code]
        now = time.time()
        fid = self.db.open_fault(now, st.node_id or None, code, name, severity, detail)
        st.faults[code] = {"id": fid, "detail": detail, "ts": now}
        self.metrics.inc_faults(1)
        self.db.insert_system_event(
            "fault_opened", f"FAULT {code} {name} node={st.node_id or '-'}: {detail}", severity)
        node_label = str(st.node_id) if st.node_id else "-"
        log.warning("FAULT %d %s node=%s: %s", code, name, node_label, detail)
        self.pub(config.T_FAULT, {
            "id": fid, "ts": round(now, 3), "fault_code": code, "fault": name, "severity": severity,
            "node": st.node_id or None, "state": "ACTIVE", "detail": detail}, qos=1)

    def _clear_fault(self, st, code, detail):
        info = st.faults.pop(code, None)
        if not info:
            return
        now = time.time()
        self.db.clear_fault(info["id"], now)
        name, severity = config.FAULTS[code]
        self.db.insert_system_event(
            "fault_cleared", f"FAULT {code} {name} node={st.node_id or '-'} CLEARED: {detail}", "INFO")
        node_label = str(st.node_id) if st.node_id else "-"
        log.info("FAULT %d %s node=%s CLEARED after %.0f s", code, name, node_label, now - info["ts"])
        self.pub(config.T_FAULT, {
            "id": info["id"], "ts": round(now, 3), "fault_code": code, "fault": name, "severity": severity,
            "node": st.node_id or None, "state": "CLEARED", "detail": detail,
            "duration_s": round(now - info["ts"], 1)}, qos=1)

    # =====================================================================
    # node status (deterministic timeout logic)
    # =====================================================================
    def _set_status(self, st, new, reason):
        if st.status == new:
            return
        old, st.status, st.status_since = st.status, new, time.time()
        self.db.insert_status(st.status_since, st.node_id, old, new, reason)
        log.info("node%d %s -> %s (%s)", st.node_id, old, new, reason)
        if new == ONLINE and 2 in st.faults:
            self._clear_fault(st, 2, "packets received again")
        self.publish_status(st)

    def publish_status(self, st):
        now = time.time()
        total = st.packets + st.lost
        self.pub(config.t_status(st.node_id), {
            "node": st.node_id, "status": st.status, "since": round(st.status_since, 3),
            "last_seen": round(st.last_rx_ts, 3) if st.last_rx_ts else None,
            "age_s": round(now - st.last_rx_ts, 1) if st.last_rx_ts else None,
            "last_seq": st.last_seq, "rssi": st.rssi,
            "packets": st.packets, "lost": st.lost, "duplicates": st.dup, "out_of_order": st.ooo,
            "delayed": st.delayed, "reboots": st.reboots,
            "loss_pct": round(100.0 * st.lost / total, 1) if total else 0.0,
            "route": {"via": st.route[0], "hops": st.route[1]},
            "sensors": st.sensor_state,
            "active_faults": sorted(st.faults.keys()),
            "last": {k: st.last_payload.get(k) for k in ("temp", "hum", "light")},
        }, qos=1, retain=True)

    def watchdog_tick(self):
        """Runs every second: derive ONLINE / STALE / OFFLINE from packet age."""
        now = time.time()
        with self.lock:
            self.release_pending(now)
            in_grace = ((now - self.started) < config.STARTUP_GRACE_S
                        or (now - self.coord_back_ts) < config.NODE_TIMEOUT_S)
            coordinator_down = self.coordinator_online is False or self.bridge_online is False
            for st in self.nodes.values():
                if not st.last_rx_ts:
                    continue
                age = now - st.last_rx_ts
                if coordinator_down:
                    # We cannot see the nodes at all: this is FAULT 5, not a node failure.
                    if st.status != UNKNOWN:
                        self._set_status(st, UNKNOWN, "coordinator/bridge down - node state not observable")
                    continue
                if age >= config.NODE_TIMEOUT_S:
                    if st.status != OFFLINE and not in_grace:
                        self._set_status(st, OFFLINE, f"no packet for {age:.0f} s")
                        self._open_fault(st, 2, f"no packets for {age:.0f} s while coordinator is alive; "
                                                "cause (link vs power vs crash) not determined by passive monitoring")
                elif age >= config.NODE_STALE_S:
                    if st.status == ONLINE:
                        self._set_status(st, STALE, f"{age:.0f} s since last packet")

    def publish_loop(self):
        last_status = last_summary = last_stats = last_purge = last_metrics = 0.0
        while not self.stop.wait(1.0):
            self.watchdog_tick()
            now = time.time()
            with self.lock:
                if now - last_metrics >= 5.0:
                    try:
                        self.db.save_system_metrics(self.metrics.snapshot())
                    except Exception as exc:
                        log.debug("metrics save error: %s", exc)
                    last_metrics = now
                if now - last_status >= config.STATUS_REFRESH_S:
                    for st in self.nodes.values():
                        self.publish_status(st)
                    last_status = now
                if now - last_summary >= config.SUMMARY_PUBLISH_S:
                    self.publish_summary()
                    last_summary = now
                if now - last_stats >= 30:
                    self.publish_stats()
                    last_stats = now
                if now - last_purge >= 3600:
                    self.db.purge_older_than(config.RETENTION_DAYS)
                    last_purge = now

    def publish_summary(self):
        total = len(self.nodes)
        counts = {s: sum(1 for n in self.nodes.values() if n.status == s) for s in (ONLINE, STALE, OFFLINE, UNKNOWN)}
        active = []
        for st in list(self.nodes.values()) + [self._system_state()]:
            for code, info in st.faults.items():
                active.append({"node": st.node_id or None, "fault_code": code,
                               "fault": config.FAULTS[code][0], "since": round(info["ts"], 3)})
        health = 100.0 * (counts[ONLINE] + 0.5 * counts[STALE]) / total if total else 0.0
        if self.coordinator_online is False or self.bridge_online is False:
            health = 0.0
        node_faults = sum(1 for a in active if a["node"] and a["fault_code"] in (1, 7))
        health = max(0.0, health - 5.0 * node_faults)
        self.pub(config.T_SUMMARY, {
            "ts": round(time.time(), 3), "total_nodes": total, "online": counts[ONLINE], "stale": counts[STALE],
            "offline": counts[OFFLINE], "unknown": counts[UNKNOWN], "recovering": 0,
            "coordinator": ("online" if self.coordinator_online else
                            "offline" if self.coordinator_online is False else "unknown"),
            "active_fault_count": len(active), "active_faults": active,
            "network_health_pct": round(health, 1)}, qos=1, retain=True)

    def publish_stats(self):
        snap = self.metrics.snapshot()
        try:
            self.db.save_system_metrics(snap)
        except Exception as exc:
            log.debug("metrics save error: %s", exc)
        self.pub(config.T_STATS, {
            "ts": round(time.time(), 3), "uptime_s": round(time.time() - self.started),
            "counters": self.counters, "db": self.db.counts(), "ml": self.engine.summary(),
            "edge_metrics": snap,
        }, qos=0)

    # =====================================================================
    def run(self):
        connect_async(self.client)
        threading.Thread(target=self.publish_loop, daemon=True).start()
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        log.info("processor running - subscribing to %s", ", ".join(config.t_data(n) for n in config.NODE_IDS))
        while not self.stop.is_set():
            time.sleep(0.3)
        log.info("shutting down")
        publish_json(self.client, config.T_BACKEND, {"status": "offline", "since": time.time()}, qos=1, retain=True)
        time.sleep(0.3)
        self.client.loop_stop()
        self.client.disconnect()
        self.db.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    Processor().run()


if __name__ == "__main__":
    main()
