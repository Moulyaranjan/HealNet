"""
Serial line protocol (coordinator -> laptop) and its translation to MQTT.

The coordinator prints ONE JSON object per line over USB serial. Lines that are
not JSON (boot logs, debug prints) are ignored, so Serial.println() debugging in
the firmware does not break the bridge.

Line types
----------
data   : {"type":"data","node":1,"seq":42,"temp":27.0,"hum":55.0,"light":1830,
          "valid":true,"dht_ok":true,"ldr_ok":true,"rssi":-48,"uptime_ms":84000,
          "boot":3,"via":null,"hops":0}
         Required: type, node, seq. Everything else is optional; the backend
         copes with missing fields. "via"/"hops" are for future multi-hop use.
hb     : {"type":"hb","uptime_ms":123456,"nodes":{"1":"ONLINE","2":"OFFLINE"}}
event  : {"type":"event","event":"node_timeout","node":1,"detail":"no packet 10s"}

The same envelope is what the coordinator would publish if it is later moved to
direct Wi-Fi MQTT, so the backend does not change when the bridge is retired.
"""
import json
import math

import config

SCHEMA = 1


def adapt_sensor_payload(msg):
    """
    Adapter that normalizes coordinator sensor payloads into the backend internal format.
    Accepts both:
      - Current coordinator format:
        {"type": "sensor", "node_id": 1, "temperature": 29.5, "humidity": 45.4,
         "light": 834, "sequence": 317, "uptime_ms": 155476}
      - Internal/legacy backend format:
        {"type": "data", "node": 1, "temp": 29.5, "hum": 45.4, "light": 834,
         "seq": 317, "uptime_ms": 155476}
      - Mixed/extended formats with sequence_number, temp, humidity, etc.
    Preserves timestamps (timestamp, ts, rx_ts) and optional metadata (boot, rssi, via, hops, valid, etc.).
    """
    if not isinstance(msg, dict):
        return msg

    out = dict(msg)

    # Normalize kind/type: "sensor" / "sensor_data" -> "data"
    kind = out.get("type")
    if kind in ("sensor", "sensor_data"):
        out["type"] = "data"

    # node mapping: node_id -> node
    if out.get("node") is None:
        for k in ("node_id", "nodeId", "id"):
            if k in out and out[k] is not None:
                out["node"] = out[k]
                break

    # sequence mapping: sequence / sequence_number -> seq
    if out.get("seq") is None:
        for k in ("sequence", "sequence_number", "seq_num", "sequenceNumber"):
            if k in out and out[k] is not None:
                out["seq"] = out[k]
                break

    # temperature mapping: temperature -> temp
    if out.get("temp") is None:
        for k in ("temperature", "temp_c", "temperature_c"):
            if k in out and out[k] is not None:
                out["temp"] = out[k]
                break

    # humidity mapping: humidity -> hum
    if out.get("hum") is None:
        for k in ("humidity", "hum_pct", "humidity_pct"):
            if k in out and out[k] is not None:
                out["hum"] = out[k]
                break

    # light mapping
    if out.get("light") is None:
        for k in ("light_intensity", "ldr", "lux", "adc"):
            if k in out and out[k] is not None:
                out["light"] = out[k]
                break

    # uptime_ms mapping
    if out.get("uptime_ms") is None:
        for k in ("uptime", "uptimeMs", "millis"):
            if k in out and out[k] is not None:
                out["uptime_ms"] = out[k]
                break

    # routing & forwarder metadata mapping
    if out.get("forwarded_by") is None:
        for k in ("forwarder", "fwd_by", "fwd"):
            if k in out and out[k] is not None:
                out["forwarded_by"] = out[k]
                break

    if out.get("source") is None:
        for k in ("src", "source_node", "origin"):
            if k in out and out[k] is not None:
                out["source"] = out[k]
                break

    if "via" not in out:
        for k in ("route_via", "hop_via", "forwarded_via"):
            if k in out:
                out["via"] = out[k]
                break

    if "hops" not in out:
        for k in ("hop_count", "num_hops"):
            if k in out and out[k] is not None:
                out["hops"] = out[k]
                break

    # timestamp mapping: if rx_ts is not present, check ts or timestamp
    if out.get("rx_ts") is None:
        for k in ("ts", "timestamp", "time"):
            if k in out and out[k] is not None:
                out["rx_ts"] = out[k]
                break

    return out


def parse_line(raw):
    """Return a dict for a valid HEALNET serial line, else None."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    raw = raw.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return None
    try:
        msg = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(msg, dict):
        return None
    if "type" not in msg:
        if "node_id" in msg or "node" in msg:
            msg["type"] = "sensor"
        elif "status" in msg or "uptime_ms" in msg:
            msg["type"] = "hb"
        elif "event" in msg:
            msg["type"] = "event"
        else:
            return None
    return msg


def _clean(value):
    """NaN/Infinity are not valid JSON for other consumers (Node-RED, browsers): send null instead."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def translate(msg, rx_ts, bridge_seq):
    """
    Convert one serial message into a list of MQTT publications:
        [(topic, payload_dict, qos, retain), ...]
    Sensor data is QoS 1 so nothing is lost across a broker reconnect;
    duplicates that result are removed by the processor.
    """
    adapted = adapt_sensor_payload(msg)
    kind = adapted.get("type")

    if kind == "data":
        node = adapted.get("node")
        if not isinstance(node, int) or isinstance(node, bool):
            return []
        envelope = {k: _clean(v) for k, v in adapted.items()}
        envelope.pop("type", None)
        envelope.update({
            "schema": SCHEMA,
            "rx_ts": round(rx_ts, 3),        # authoritative timestamp: laptop clock at arrival
            "bridge_seq": bridge_seq,
        })
        # Preserve source node name if present in adapted message, else default to "serial"
        envelope["source"] = adapted.get("source") or "serial"
        envelope["bridge_source"] = "serial"
        return [(config.t_data(node), envelope, 1, False)]

    if kind in ("event", "status"):
        payload = dict(msg)
        payload.pop("type", None)
        payload["rx_ts"] = round(rx_ts, 3)
        return [(config.T_COORD_EVENT, payload, 1, False)]

    # "hb" lines only refresh coordinator liveness; the bridge publishes the
    # consolidated coordinator status itself, so nothing is forwarded here.
    return []
