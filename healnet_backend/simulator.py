"""
HEALNET coordinator simulator - test the whole backend without any hardware.

It behaves like the coordinator ESP32: two sensor nodes, one packet each every
2 s, plus coordinator heartbeats. Faults are injected on a timeline.

Two output modes
  --mode mqtt    (default) publish exactly what serial_bridge.py would publish
  --mode stdout  print the coordinator's serial JSON lines, so you can test the
                 bridge too:   python simulator.py --mode stdout | python serial_bridge.py --stdin

Scripted scenarios:   --events "20:n1_off,40:n1_on,60:dht1_fail,80:dht1_ok"
Built-in demo:        --demo     (runs every scenario in sequence, ~5 minutes)

Event commands (replace 1 with 2 for the other node)
  n1_off / n1_on         node stops / resumes transmitting (power or link loss)
  dht1_fail / dht1_ok    DHT11 returns invalid readings
  ldr1_stuck / ldr1_ok   LDR output pinned at the ADC rail
  spike1                 one-off temperature spike on node 1 only (sensor suspect)
  env_light              room light change seen by BOTH nodes (environmental)
  dup1 / ooo1            duplicate packet / one packet delivered late (out-of-order) from node 1
  loss1_30 / loss1_0     random packet loss on node 1 (30 % / none)
  reboot1                node 1 restarts (seq and uptime reset)
  coord_off / coord_on   coordinator USB link lost / restored
  via1_2 / via1_direct   node 1 reported as forwarded through node 2 / direct (future multi-hop demo)
"""
import argparse
import json
import math
import random
import signal
import sys
import time

import config
import protocol

DEMO = ("15:dht1_fail,30:dht1_ok,40:spike1,55:env_light,75:n2_off,95:n2_on,"
        "110:dup1,112:ooo1,120:loss1_30,140:loss1_0,150:ldr2_stuck,185:ldr2_ok,"
        "195:reboot2,210:via1_2,225:via1_direct,240:coord_off,255:coord_on")


class SimNode:
    def __init__(self, node_id, phase):
        self.id = node_id
        self.phase = phase
        self.seq = 0
        self.boot = 1
        self.t0 = time.time()
        self.online = True
        self.dht_fail = False
        self.ldr_stuck = False
        self.loss = 0.0
        self.via = None
        self.hops = 0
        self.spike = 0
        self.last = None
        self.hold_next = False
        self.held = None

    def reading(self, now, env_light_offset):
        t = now - self.t0
        temp = 26.0 + 1.2 * math.sin(t / 90 + self.phase) + random.gauss(0, 0.15)
        hum = 58.0 + 3.0 * math.sin(t / 120 + self.phase * 2) + random.gauss(0, 0.5)
        light = 1800 + 250 * math.sin(t / 60 + self.phase) + random.gauss(0, 15) + env_light_offset
        if self.spike:
            temp += 14.0
            self.spike -= 1
        dht_ok = not self.dht_fail
        msg = {
            "type": "data", "node": self.id, "seq": self.seq, "boot": self.boot,
            "temp": round(round(temp), 1) if dht_ok else float("nan"),
            "hum": round(round(hum), 1) if dht_ok else float("nan"),
            "light": 4095 if self.ldr_stuck else int(max(0, min(4095, light))),
            "valid": dht_ok, "dht_ok": dht_ok, "ldr_ok": True,
            "rssi": random.randint(-62, -44),
            "uptime_ms": int((now - self.t0) * 1000),
            "via": self.via, "hops": self.hops,
        }
        self.seq += 1
        return msg


class Simulator:
    def __init__(self, mode, events, duration=None):
        self.duration = duration
        self.mode = mode
        self.nodes = {1: SimNode(1, 0.0), 2: SimNode(2, 1.3)}
        self.events = sorted(events)
        self.start = time.time()
        self.coord_up = True
        self.env_until = 0.0
        self.bridge_seq = 0
        self.stop = False
        self.client = None
        if mode == "mqtt":
            from mqtt_utils import connect_async, make_client
            self.client = make_client("healnet-simulator")
            connect_async(self.client)
            time.sleep(1.0)

    # -- output -------------------------------------------------------------
    def emit(self, msg):
        if self.mode == "stdout":
            print(json.dumps(msg), flush=True)
            return
        from mqtt_utils import publish_json
        self.bridge_seq += 1
        for topic, payload, qos, retain in protocol.translate(msg, time.time(), self.bridge_seq):
            publish_json(self.client, topic, payload, qos, retain)

    def coordinator_status(self):
        if self.mode != "mqtt":
            return
        from mqtt_utils import publish_json
        publish_json(self.client, config.T_COORDINATOR, {
            "status": "online" if self.coord_up else "offline", "ts": round(time.time(), 3),
            "last_line_age_s": 0.0 if self.coord_up else 30.0, "port": "SIMULATED"}, qos=1, retain=True)

    # -- events ---------------------------------------------------------------
    def apply(self, cmd):
        print(f"[sim t={time.time() - self.start:5.1f}s] >>> {cmd}", file=sys.stderr, flush=True)
        if cmd == "env_light":
            self.env_until = time.time() + 12
            return
        if cmd == "coord_off":
            self.coord_up = False
            return
        if cmd == "coord_on":
            self.coord_up = True
            return
        name, _, arg = cmd.partition("_")
        digit = next((c for c in name if c.isdigit()), None)
        if digit is None:
            print(f"unknown command {cmd}", file=sys.stderr)
            return
        n = self.nodes[int(digit)]
        base = name.rstrip("0123456789")
        if base == "n" and arg == "off":
            n.online = False
        elif base == "n" and arg == "on":
            n.online = True
        elif base == "dht":
            n.dht_fail = arg == "fail"
        elif base == "ldr":
            n.ldr_stuck = arg == "stuck"
        elif base == "spike":
            n.spike = 3
        elif base == "dup" and n.last:
            self.emit(dict(n.last))
        elif base == "ooo":
            n.hold_next = True                      # next packet is delayed and arrives after the one after it
        elif base == "loss":
            n.loss = int(arg) / 100.0
        elif base == "reboot":
            n.seq, n.boot, n.t0 = 0, n.boot + 1, time.time()
        elif base == "via":
            n.via, n.hops = (2, 1) if arg == "2" else ((1, 1) if arg == "1" else (None, 0))
        else:
            print(f"unknown command {cmd}", file=sys.stderr)

    # -- main loop ------------------------------------------------------------
    def run(self):
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        next_tick = time.time()
        while not self.stop:
            now = time.time()
            elapsed = now - self.start
            if self.duration and elapsed >= self.duration:
                break
            while self.events and self.events[0][0] <= elapsed:
                self.apply(self.events.pop(0)[1])
            self.coordinator_status()
            if self.coord_up:
                env = -700.0 if now < self.env_until else 0.0
                for n in self.nodes.values():
                    if not n.online:
                        continue
                    msg = n.reading(now, env)
                    n.last = msg
                    if random.random() < n.loss:
                        continue                       # packet lost over the air: seq is consumed
                    if n.hold_next:
                        n.held, n.hold_next = msg, False
                        continue
                    self.emit(msg)
                    if n.held:
                        self.emit(n.held)              # arrives late => out-of-order at the coordinator
                        n.held = None
                if self.mode == "stdout":
                    print(json.dumps({"type": "hb", "uptime_ms": int(elapsed * 1000)}), flush=True)
            next_tick += config.SENSOR_INTERVAL_S
            time.sleep(max(0.0, next_tick - time.time()))
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


def parse_events(text):
    out = []
    for item in filter(None, (text or "").split(",")):
        t, _, cmd = item.partition(":")
        out.append((float(t), cmd.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description="HEALNET coordinator simulator")
    ap.add_argument("--mode", choices=("mqtt", "stdout"), default="mqtt")
    ap.add_argument("--events", default="", help='e.g. "20:n1_off,40:n1_on"')
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--demo", action="store_true", help="run the built-in scenario timeline")
    args = ap.parse_args()
    Simulator(args.mode, parse_events(DEMO if args.demo else args.events), args.duration).run()


if __name__ == "__main__":
    main()
