"""
HEALNET serial -> MQTT bridge.

    Coordinator ESP32 --USB serial (JSON lines)--> THIS BRIDGE --MQTT--> Mosquitto

Responsibilities
  * read JSON lines from the coordinator (non-JSON debug output is ignored),
  * stamp every packet with the laptop receive time (ESP32 has no real clock),
  * publish sensor packets to healnet/nodeN/data (QoS 1),
  * publish the coordinator's own liveness to healnet/system/coordinator (retained),
    so the backend can tell "coordinator/USB is down" apart from "a sensor node is down",
  * keep data when the broker is unreachable: packets are spooled to disk and
    replayed in order when the broker returns (the processor removes duplicates).

Usage
    python serial_bridge.py                    # auto-detect the serial port
    python serial_bridge.py --port COM5        # Windows
    python serial_bridge.py --port /dev/ttyUSB0
    python simulator.py --stdout | python serial_bridge.py --stdin     # no hardware
"""
import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

import config
import protocol
from mqtt_utils import connect_async, make_client, publish_json

log = logging.getLogger("healnet.bridge")


def find_serial_port():
    from serial.tools import list_ports
    wanted_vids = {0x10C4, 0x1A86, 0x0403, 0x303A}        # CP210x, CH340, FTDI, Espressif native USB
    for p in list_ports.comports():
        if p.vid in wanted_vids or "USB" in (p.description or "").upper():
            return p.device
    return None


class Spool:
    """Append-only JSONL file used while the broker is unreachable."""

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, "bridge_spool.jsonl")
        self.lock = threading.Lock()
        self.count = self._count_existing()
        self.dropped = 0

    def _count_existing(self):
        if not os.path.exists(self.path):
            return 0
        with open(self.path, "rb") as fh:
            return sum(1 for _ in fh)

    def add(self, topic, payload, qos):
        with self.lock:
            if self.count >= config.SPOOL_MAX_LINES:
                self.dropped += 1
                return False
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": topic, "p": payload, "q": qos}) + "\n")
            self.count += 1
            return True

    def drain(self):
        """Atomically take everything currently in the spool."""
        with self.lock:
            if not os.path.exists(self.path):
                return []
            tmp = self.path + ".replay"
            os.replace(self.path, tmp)
            self.count = 0
        items = []
        with open(tmp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    items.append(json.loads(line))
                except ValueError:
                    continue
        os.remove(tmp)
        return items


class Bridge:
    def __init__(self):
        self.client = make_client("healnet-bridge", config.T_BRIDGE, {"status": "offline"})
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.spool = Spool(config.SPOOL_DIR)
        self.stop = threading.Event()
        self.bridge_seq = 0
        self.lines = 0
        self.bad_lines = 0
        self.last_line_ts = 0.0
        self.last_hb = None
        self.port_name = None
        self.last_info = None
        self._replaying = threading.Lock()

    # ---- MQTT callbacks ---------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connect refused: %s", reason_code)
            return
        log.info("MQTT connected to %s:%s", config.MQTT_HOST, config.MQTT_PORT)
        publish_json(client, config.T_BRIDGE, {"status": "online", "since": time.time()}, qos=1, retain=True)
        if self.spool.count:
            threading.Thread(target=self._replay_spool, daemon=True).start()

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        log.warning("MQTT disconnected (%s) - packets will be spooled to disk until it returns", reason_code)

    def _replay_spool(self):
        if not self._replaying.acquire(blocking=False):
            return
        try:
            items = self.spool.drain()
            if items:
                log.info("replaying %d spooled messages", len(items))
            for i, it in enumerate(items):
                if not self.client.is_connected():          # broker dropped again mid-replay
                    for rest in items[i:]:
                        self.spool.add(rest["t"], rest["p"], rest["q"])
                    return
                publish_json(self.client, it["t"], it["p"], qos=it["q"])
                time.sleep(0.002)                            # gentle pacing for the broker
        finally:
            self._replaying.release()

    # ---- publishing -------------------------------------------------------
    def publish(self, topic, payload, qos=1, retain=False):
        if self.client.is_connected():
            ok = publish_json(self.client, topic, payload, qos, retain)
            if ok:
                return
        if qos >= 1 and not retain:                          # only real data is worth spooling
            self.spool.add(topic, payload, qos)

    # ---- serial lines -----------------------------------------------------
    def handle_line(self, raw):
        msg = protocol.parse_line(raw)
        now = time.time()
        if msg is None:
            if raw.strip():
                self.bad_lines += 1
                log.debug("ignored non-JSON line: %r", raw[:80])
            return
        self.lines += 1
        self.last_line_ts = now
        if msg.get("type") in ("hb", "heartbeat") or (msg.get("type") == "status" and "node_id" not in msg and "node" not in msg):
            self.last_hb = msg
            return
        self.bridge_seq += 1
        for topic, payload, qos, retain in protocol.translate(msg, now, self.bridge_seq):
            self.publish(topic, payload, qos, retain)

    def read_serial(self, port, baud):
        import serial
        while not self.stop.is_set():
            name = port or find_serial_port()
            if not name:
                log.warning("no serial port found - plug in the coordinator (retrying)")
                self.stop.wait(3)
                continue
            try:
                with serial.Serial(name, baud, timeout=1) as ser:
                    self.port_name = name
                    log.info("reading coordinator on %s @ %d baud", name, baud)
                    ser.reset_input_buffer()
                    while not self.stop.is_set():
                        raw = ser.readline()
                        if raw:
                            self.handle_line(raw.decode("utf-8", errors="ignore"))
            except (serial.SerialException, OSError) as exc:
                log.warning("serial error on %s: %s - reconnecting in 2 s", name, exc)
                self.port_name = None
                self.stop.wait(2)

    def read_stdin(self):
        for raw in sys.stdin:
            if self.stop.is_set():
                break
            self.handle_line(raw)
        log.info("stdin closed")

    # ---- coordinator liveness ---------------------------------------------
    def coordinator_status_loop(self):
        last_state = None
        while not self.stop.wait(config.COORDINATOR_PUBLISH_S):
            age = time.time() - self.last_line_ts if self.last_line_ts else None
            online = age is not None and age < config.COORDINATOR_TIMEOUT_S
            state = "online" if online else "offline"
            if state != last_state:
                log.info("coordinator is %s", state.upper())
                last_state = state
            payload = {
                "status": state, "ts": round(time.time(), 3),
                "last_line_age_s": None if age is None else round(age, 1),
                "lines": self.lines, "bad_lines": self.bad_lines,
                "spooled": self.spool.count, "spool_dropped": self.spool.dropped,
                "port": self.port_name, "hb": self.last_hb,
            }
            if self.client.is_connected():
                publish_json(self.client, config.T_COORDINATOR, payload, qos=1, retain=True)

    # ---- lifecycle --------------------------------------------------------
    def run(self, args):
        connect_async(self.client)
        threading.Thread(target=self.coordinator_status_loop, daemon=True).start()
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        if args.stdin:
            reader = threading.Thread(target=self.read_stdin, daemon=True)
        else:
            reader = threading.Thread(target=self.read_serial, args=(args.port or config.SERIAL_PORT, args.baud),
                                      daemon=True)
        reader.start()
        while not self.stop.is_set() and reader.is_alive():
            time.sleep(0.2)
        if args.stdin:                                        # let in-flight messages leave
            deadline = time.time() + 3
            while self.client.is_connected() and time.time() < deadline and self.spool.count:
                time.sleep(0.1)
            time.sleep(0.5)
        publish_json(self.client, config.T_BRIDGE, {"status": "offline", "since": time.time()}, qos=1, retain=True)
        time.sleep(0.3)
        self.client.loop_stop()
        self.client.disconnect()


def _acquire_instance_lock(lock_path):
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r") as f:
                old_pid = int(f.read().strip())
            import psutil
            if psutil.pid_exists(old_pid):
                return False, old_pid
        except Exception:
            pass
    try:
        with open(lock_path, "w") as f:
            f.write(str(os.getpid()))
        return True, os.getpid()
    except Exception as exc:
        log.warning("Could not write lock file: %s", exc)
        return True, os.getpid()


def _release_instance_lock(lock_path):
    try:
        if os.path.exists(lock_path):
            with open(lock_path, "r") as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                os.remove(lock_path)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="HEALNET serial -> MQTT bridge")
    ap.add_argument("--port", default="", help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=config.SERIAL_BAUD)
    ap.add_argument("--stdin", action="store_true", help="read JSON lines from stdin instead of a serial port")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    lock_path = os.path.join(config.SPOOL_DIR, "serial_bridge.lock")
    acquired, pid = _acquire_instance_lock(lock_path)
    if not acquired:
        log.error("Another serial_bridge instance is already running (PID %d). Only one bridge may run to prevent serial conflicts.", pid)
        sys.exit(1)
    try:
        Bridge().run(args)
    finally:
        _release_instance_lock(lock_path)


if __name__ == "__main__":
    main()
