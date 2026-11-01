# HEALNET – MQTT data collection & processing subsystem

Scope of this package: **sensor packet → MQTT → validation → fault/anomaly analysis → database → live topics for Node-RED / any dashboard.**
Recovery, relay control and multi-hop routing are *not* implemented here; the topics, tables and packet fields they will need are reserved so they plug in later without changes to this code.

```
Sensor Node 1 ─┐  ESP-NOW                USB serial            MQTT (QoS 1)
               ├──────────▶ Coordinator ──(JSON lines)──▶ serial_bridge.py ──▶ Mosquitto ──▶ processor.py ──▶ SQLite
Sensor Node 2 ─┘                ESP32                                           │   │                │
                                                                                │   └─▶ monitor.py   └─▶ healnet/… processed / status / fault /
                                                                                └─▶ Node-RED                  anomaly / routing / summary
```

## Why serial bridge (recommended for the current prototype)

| | USB serial → bridge (**chosen**) | Coordinator publishes MQTT over Wi-Fi |
|---|---|---|
| Wi-Fi channel | ESP-NOW needs the radio on **channel 1**; joining your router (usually another channel) forces the coordinator onto *that* channel, so the sensor nodes would have to be re-configured to match | – |
| Firmware change | print one JSON line | Wi-Fi + MQTT client + reconnect logic on the coordinator |
| Debugging | Serial Monitor shows exactly what is sent | harder |
| Buffering when broker is down | bridge spools to disk | would need flash/RAM buffering on the ESP32 |
| Coordinator failure vs broker failure | distinguishable (bridge reports coordinator liveness) | harder |

The message format is identical either way, so moving to direct Wi-Fi MQTT later means retiring `serial_bridge.py` only.

## Quick start

```bash
pip install -r requirements.txt
mosquitto -c mosquitto.conf -v            # terminal 1  (install Mosquitto 2.x first)
python processor.py                       # terminal 2  backend processor
python api.py                             # terminal 3  local REST API (http://127.0.0.1:8000/docs)
python monitor.py                         # terminal 4  live MQTT monitor
python simulator.py --demo                # terminal 5  no hardware: scripted faults (~5 min)
```

With the real coordinator plugged in, replace step 5 with:

```bash
python serial_bridge.py                   # auto-detects or use --port COM14
# Note: COM14 = Coordinator ESP32. Do NOT open COM9 (Node 1) or COM11 (Node 2).
```

Offline unit tests (41 tests): `python -m unittest discover tests -v`
Train and evaluate ML models: `python ml_pipeline.py`
Windows note: the Mosquitto installer does not add itself to PATH – run it from `C:\Program Files\mosquitto`.

## Coordinator firmware format (NO FIRMWARE MODIFICATION REQUIRED)

The backend natively supports your current coordinator firmware format via a built-in compatibility adapter:

```json
{
  "type": "sensor",
  "node_id": 1,
  "temperature": 29.5,
  "humidity": 45.4,
  "light": 834,
  "sequence": 317,
  "uptime_ms": 155476
}
```

The adapter automatically maps:
- `node_id` -> `node`
- `sequence` or `sequence_number` -> `seq`
- `temperature` -> `temp`
- `humidity` -> `hum`
- `light` -> `light`
- `uptime_ms` -> `uptime_ms`
- `type`: `"sensor"` -> `"data"`

It also supports the backend's internal format:

```json
{
  "type": "data",
  "node": 1,
  "seq": 317,
  "temp": 29.5,
  "hum": 45.4,
  "light": 834,
  "uptime_ms": 155476
}
```

| Field | Required | Meaning |
|---|---|---|
| `node`, `seq` | **yes** | origin node (1/2) and its per-node sequence number |
| `temp`, `hum`, `light`, `valid` | recommended | `valid=false` is attributed to the DHT11 (an analog LDR cannot self-report) |
| `dht_ok`, `ldr_ok` | optional | explicit per-sensor health if the node firmware has it |
| `boot` | strongly recommended | increments on every node restart (e.g. kept in RTC/NVS). Without it, reboots are inferred from `uptime_ms`/`seq` going backwards |
| `uptime_ms`, `rssi` | optional | node `millis()`, ESP-NOW RSSI from the receive callback |
| `via`, `hops` | future | origin stays in `node`; `via` = forwarding node (multi-hop phase) |

Timestamps: the ESP32s have no real clock, so the **bridge stamps `rx_ts` with the laptop clock** on arrival. All backend timing uses it.

## MQTT topics

| Topic | Publisher | QoS | Retained | Content |
|---|---|---|---|---|
| `healnet/node{1,2}/data` | bridge | 1 | no | raw envelope (`rx_ts`, `bridge_seq`, fields above) |
| `healnet/node{1,2}/processed` | processor | 0 | no | validated + enriched reading: flags (`gap`, `delayed`, `out_of_order`, `reboot`), anomaly flag/score, node status |
| `healnet/node{1,2}/status` | processor | 1 | **yes** | ONLINE / STALE / OFFLINE / UNKNOWN, age, packets, lost, duplicates, loss %, reboots, route, sensor health, active fault codes, last values |
| `healnet/system/fault` | processor | 1 | no | `ACTIVE` / `CLEARED` fault events + `PACKET_LOSS` events |
| `healnet/system/anomaly` | processor | 1 | no | one message per finding with classification |
| `healnet/system/routing` | processor | 1 | no | route changes (`DIRECT` / `REROUTED`, `via`, `hops`, `path`) – fires once firmware reports `via` |
| `healnet/system/summary` | processor | 1 | **yes** | network overview: online/offline/stale counts, active faults, health % |
| `healnet/system/coordinator` | bridge | 1 | **yes** | coordinator online/offline (any serial line within 6 s), port, spool depth |
| `healnet/system/bridge`, `…/backend` | bridge / processor | 1 | **yes** | online + Last-Will `offline` |
| `healnet/system/stats` | processor | 0 | no | counters, DB sizes, ML state |
| `healnet/system/recovery` | – | 1 | – | **RESERVED** – not used in this phase |

QoS choice: sensor data and faults are **QoS 1** (never lost, possible duplicates → removed by the processor); high-rate `processed` stream is QoS 0 because the database already has the history; state topics are **retained** so a dashboard that connects late sees the current state immediately. QoS 2 is not needed – de-duplication is done by sequence number, which is cheaper and works across the serial bridge too.

## How faults are decided

Deterministic rules decide faults. ML only judges *sensor behaviour* and never marks hardware as failed.

| Fault | Raised when | Status |
|---|---|---|
| 1 SENSOR_FAILURE | DHT11 invalid/missing/out of range for 3 packets; LDR invalid for 3, or pinned at ADC rail ≥15 packets (*suspected*) | implemented |
| 2 NODE_COMM_FAILURE | no packet for 10 s **while the coordinator is alive**. Node goes STALE at 6 s, OFFLINE at 10 s; cleared by the next packet | implemented |
| 5 COORDINATOR_FAILURE | no serial lines for 6 s, or bridge offline. Nodes become **UNKNOWN**, not OFFLINE – the backend can't see them, so it does not blame them | implemented |
| 6 MQTT_BACKEND_FAILURE | broker/processor loss shows as the Last-Will on `system/backend`; the bridge spools data to disk and replays it | implemented (detection by dashboard via LWT) |
| 7 ABNORMAL_SENSOR_BEHAVIOUR | ≥5 anomalous samples in 20 s that **only this node** shows | implemented |
| 3 NODE_POWER_FAILURE, 4 WIRELESS_LINK_FAILURE | cannot be told apart from #2 by passive monitoring | **RESERVED** – needs RSSI trend, ACK/probe results, relay/current feedback |

Fault 2's detail text says explicitly that the cause (link / power / crash) is undetermined; later phases refine it.

Sequence analysis per node: duplicates dropped, gaps counted as lost packets (a `PACKET_LOSS` event if ≥3 missing), late packets flagged `out_of_order`, restarts detected via `boot` / uptime / sequence regression, packets older than 6 s on arrival flagged `delayed` (stored as history, but they never change live node status).

## Machine learning (anomaly.py)

Per node, on every valid sample: **median/MAD robust z-score** against the node's own 10-minute window → **rate-of-change limit** → **IsolationForest** on `[temp, hum, light, Δtemp, Δhum, Δlight]` (trained after 120 samples, retrained every 150; counted only when another indicator is mildly elevated, to keep false alarms low). Then the cross-node check, done ~2.8 s later so the other node's sample has arrived:

* both nodes moved the same way → `environmental_change` (e.g. room light switched on)
* only this node moved → `localized_sensor_suspect`
* other node has no data nearby → `unconfirmed`

A one-off spike is published as an anomaly but only *sustained* single-node deviation raises FAULT 7. Random Forest is not included: it needs labelled fault data – collect it from `anomalies`/`faults` tables later.

## Reliability

* Processor uses a fixed client id with `clean_session=false` and QoS 1 subscriptions → the broker queues sensor data while the backend is down (see `max_queued_messages` in `mosquitto.conf`).
* Bridge → broker outage: packets spooled to `spool/bridge_spool.jsonl` (cap 50 000), replayed in order on reconnect.
* History survives restarts: ML windows and sequence state are reloaded from SQLite; the startup grace period (12 s) prevents false timeout faults.

## Files

* `config.py` — Centralized settings (overridable with `HEALNET_*` env vars).
* `protocol.py` — Serial line decoder, format compatibility adapter (`adapt_sensor_payload`), MQTT translation.
* `serial_bridge.py` — USB serial to MQTT bridge with disk spooling and coordinator liveness tracking.
* `processor.py` — Primary backend stream processor: validation, fault detection, ML integration, SQLite writes.
* `storage.py` — SQLite database engine with WAL mode: readings, status events, faults, anomalies, route events, system events, recovery events.
* `anomaly.py` — Real-time sensor anomaly detection: Median/MAD, rate-of-change, multivariate Isolation Forest, cross-node correlation.
* `ml_pipeline.py` — Offline ML training, model evaluation (IsolationForest, LOF, One-Class SVM), metrics reporting, model persistence (`models/`).
* `edge_metrics.py` — Edge computing resource tracking (CPU, RAM, disk via psutil) and processing latency measurement.
* `api.py` — Local FastAPI REST server exposing sensor readings, node health, faults, anomalies, edge metrics, and ML metadata.
* `simulator.py` — Coordinator and sensor node simulator for testing without physical hardware.
* `monitor.py` — Live terminal subscriber for all MQTT topics.
* `mosquitto.conf` — Local Mosquitto 2.x broker configuration.
* `tests/test_backend.py` — Core protocol, sequence, fault detection, and coordinator format unit tests.
* `tests/test_full_system.py` — ML pipeline, edge metrics, FastAPI endpoints, and recovery event integration tests.

## Node-RED

MQTT-in nodes on `healnet/node1/processed`, `healnet/node2/processed` (graphs), `healnet/+/status` and `healnet/system/summary` (tiles; retained, so they fill instantly), `healnet/system/fault` and `healnet/system/anomaly` (event log). Payloads are plain JSON – set the node output to "parsed JSON object".

## Status & limitations

* **Implemented and tested here with a simulator and a stand-in broker** (Mosquitto itself was not available in the test environment): validation, de-duplication, gap/OOO/reboot detection, node timeouts, FAULT 1/2/5/7, anomaly classification, SQLite, spool/replay, retained status/summary. Nothing has been run against your real ESP32s yet – expect to adjust thresholds (`config.py`) after you see real DHT11/LDR data.
* Passive monitoring cannot separate a dead node from a dead radio link (faults 3/4).
* ESP32 ADC pins and a DHT11 are noisy: `RATE_LIMITS`, `MAD_FLOOR` and the LDR rail values are starting points.
* A `valid=false` without per-sensor flags is attributed to the DHT11.
* The simulator's multi-hop (`via`) packets only exercise the *reporting* path; no forwarding exists in firmware yet.
