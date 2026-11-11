# HEALNET — Coordinator-Centric Self-Healing IoT Sensor Network

HEALNET is a resilient, self-healing IoT sensor network and anomaly-detection backend for ESP32-based wireless nodes. It implements a **Coordinator-Centric Architecture** with **Sensor-Level Failover via CD4053 Multiplexing**, deterministic fault detection, multivariate unsupervised machine learning for sensor anomalies, and strict automatic recovery verification.

> **Core System Principle:**  
> *"Node failure does not necessarily mean sensor data loss. HEALNET maintains physical sensor data availability through an alternate node via CD4053 hardware multiplexer and verifies automatic recovery when the failed node returns."*

---

## 1. System Architecture

### Normal Operation vs Hardware Failover Operation

```
Normal Operation:
Node 1 Sensors ──▶ Node 1 ESP32 ──ESP-NOW──┐
                                           │
                                           ▼
                                      Coordinator ESP32 (Single USB Serial)
                                           ▲
                                           │
Node 2 Sensors ──▶ Node 2 ESP32 ──ESP-NOW──┘
                                           │
                                       USB Serial (115200 Baud)
                                           │
                                           ▼
                                   serial_bridge.py
                                           │
                                           ▼ MQTT (1883)
                                      processor.py ────▶ SQLite (healnet.db)
                                           │
                                           ├───────────▶ Local REST API (FastAPI :8000)
                                           ▼
                                      Web Dashboard (healnet_dashboard.html)

Hardware Failover Operation (Node 1 ESP32 Fails):
Node 1 ESP32 [FAILED / OFFLINE]
Node 1 Sensors ──▶ CD4053 MUX ──▶ Node 2 ESP32 ──ESP-NOW (node=1, via=2, hops=1) ──┐
                                                                                   │
Node 2 Sensors ─────────────────▶ Node 2 ESP32 ──ESP-NOW (node=2, via=null, hops=0)─┤
                                                                                   ▼
                                                                     Coordinator ESP32
                                                                                   │
                                                                               USB Serial
                                                                                   │
                                                                                   ▼
                                                                             Python Backend
```

### Coordinator-Centric Serial Architecture
* **Single Serial Connection**: The Coordinator ESP32 is the **only** hardware device directly connected to the host laptop/backend via USB Serial (`serial_bridge.py`).
* Sensor Node 1 and Sensor Node 2 communicate **exclusively wirelessly** with the Coordinator via ESP-NOW.
* The backend does **not** establish independent serial connections to Node 1 or Node 2, eliminating outdated multi-COM-port dependencies.
* All telemetry, heartbeats, and failover-forwarded packets from both nodes are received through the single Coordinator stream.

---

## 2. Sensor-Level Failover & Decoupled State Tracking

Conventional IoT systems mark a sensor node completely unavailable when its microcontroller stops transmitting. HEALNET decouples hardware availability from physical sensor telemetry:

| Dimension | States | Description |
|---|---|---|
| **ESP32 Status** | `ONLINE`, `OFFLINE`, `UNKNOWN` | Microcontroller wireless reachability via direct ESP-NOW packets (`via=None`, `hops=0`). |
| **Physical Sensor Status** | `AVAILABLE`, `DEGRADED`, `UNAVAILABLE` | Physical sensor hardware health (DHT11, LDR) whether read locally or via CD4053 mux. |
| **Sensor Data Status** | `FRESH`, `STALE`, `INVALID`, `MISSING` | Telemetry freshness based on timestamp and packet validity. |
| **Communication Route** | `DIRECT`, `FAILOVER VIA NODE X` | Physical routing path to the coordinator. |

### Example State Transition (Node 1 ESP32 Outage)
When Node 1 ESP32 experiences power loss or microcontroller crash:
1. Node 2 activates the CD4053 analog multiplexer to read Node 1's physical sensors.
2. Node 2 transmits packets tagged: `node=1, via=2, hops=1, forwarded_by="NODE2"`.
3. The backend evaluates:
   * **Node 1 ESP32 Status:** `OFFLINE`
   * **Node 1 Sensor Availability:** `AVAILABLE`
   * **Node 1 Data Status:** `FRESH`
   * **Node 1 Route:** `FAILOVER VIA NODE 2`
   * **Network Status:** `DEGRADED BUT OPERATIONAL`

---

## 3. Communication Protocol & Routing Metadata

### Coordinator Serial JSON Format
The USB serial bridge accepts newline-delimited JSON. Packets distinguish direct from forwarded telemetry:

**Direct Packet (Node 1 Normal):**
```json
{
  "type": "data",
  "node": 1,
  "temp": 29.2,
  "hum": 54.1,
  "light": 1741,
  "valid": true,
  "via": null,
  "hops": 0
}
```

**Forwarded Packet (Node 2 Forwarding Node 1 Sensors via CD4053):**
```json
{
  "type": "data",
  "node": 1,
  "temp": 29.2,
  "hum": 54.1,
  "light": 1725,
  "valid": true,
  "via": 2,
  "hops": 1,
  "forwarded_by": "NODE2",
  "source": "NODE1"
}
```

The protocol adapter (`protocol.py`) normalizes incoming packets and preserves backward compatibility with legacy coordinator envelopes.

---

## 4. 8-State Failover & Automatic Recovery Machine

HEALNET manages an 8-state transition lifecycle for each node:

1. `NORMAL`: Both nodes transmitting direct wireless packets (`via=null, hops=0`).
2. `NODE_FAILURE_DETECTED`: Direct packet timeout elapsed (>10 s) without direct packet.
3. `FAILOVER_INITIATED`: System detects failure and prepares failover route tracking.
4. `FAILOVER_ACTIVE`: Forwarded packets arriving (`via=2, hops=1`), physical sensors maintained.
5. `RECOVERY_INITIATED`: Hardware power/operation restored to failed node.
6. `RECOVERY_VALIDATION`: Coordinator awaits direct packets from the recovering node.
7. `RECOVERY_CONFIRMED`: Backend validates direct wireless packets (`via=null, hops=0`).
8. `NORMAL_RESTORED`: Route returns to `DIRECT` and state resets to `NORMAL`.

> **Strict Recovery Verification Rule:**  
> Forwarded packets from Node 2 do **NOT** count towards Node 1 recovery. Recovery is strictly confirmed only after receiving valid direct packets from Node 1's own ESP32.

---

## 5. Machine Learning & Deterministic Fault Architecture

### Deterministic Fault Engine
* **Fault 1 (`SENSOR_FAILURE`)**: Consecutive missing or out-of-range sensor readings (DHT11/LDR).
* **Fault 2 (`NODE_COMM_FAILURE`)**: Direct wireless timeout (>10 s) from node ESP32.
* **Fault 5 (`COORDINATOR_FAILURE`)**: USB serial heartbeat timeout (>6 s) from coordinator.
* **Fault 6 (`MQTT_BACKEND_FAILURE`)**: Broker disconnection / process failure.
* **Fault 7 (`ABNORMAL_SENSOR_BEHAVIOUR`)**: ML/statistical anomaly cluster detection.

### Machine Learning Isolation
* Unsupervised ML (Isolation Forest / Local Outlier Factor) inspects physical sensor readings (`temp`, `hum`, `light`) across 15 temporal features.
* **Routing Immunity**: Switching from `DIRECT` to `FAILOVER` route does not trigger false ML anomalies, as routing transitions are decoupled from sensor physics.

---

## 6. Installation & Execution

### Prerequisites
* Python 3.10 to 3.14 (tested on Windows & Linux).
* Local Mosquitto MQTT Broker (port 1883).

```bash
pip install -r requirements.txt
```

### Running the Complete Stack

**Step 1: Start Mosquitto Broker**
```bash
mosquitto -c mosquitto.conf -v
```

**Step 2: Start Edge Processor**
```bash
python processor.py
```

**Step 3: Start REST API**
```bash
python api.py
```

**Step 4: Connect Hardware Coordinator OR Run Simulator**

* **Real Hardware (Coordinator on USB Serial):**
  ```bash
  python serial_bridge.py --port COM14
  ```
  *(On Linux: `python serial_bridge.py --port /dev/ttyUSB0`)*

* **Software Simulation (Full Failover & Recovery Testing):**
  ```bash
  python simulator.py --events "10:failover1_2,30:recover1_direct"
  ```
  *(Or interactive mode: type `failover1_2` or `recover1_direct` directly into the simulator prompt)*

**Step 5: Launch Dashboard**
Open `healnet_dashboard/healnet_dashboard.html` in any modern web browser or serve it via local HTTP server:
```bash
cd ../healnet_dashboard
python -m http.server 8080
```
Navigate to: `http://localhost:8080/healnet_dashboard.html`

---

## 7. REST API Reference

The local REST API runs at `http://127.0.0.1:8000` (Interactive docs at `http://127.0.0.1:8000/docs`).

| Endpoint | Method | Description |
|---|---|---|
| `/api/system/health` | GET | Backend health, coordinator connectivity, DB records |
| `/api/system/summary` | GET | Aggregated status: online nodes, failover nodes, network status |
| `/api/system/metrics` | GET | Edge computing CPU, RAM, disk, and processing latency |
| `/api/system/recovery` | GET | Recovery events log with duration, method, and route delta |
| `/api/nodes` | GET | Decoupled node statuses: `esp32_status`, `sensor_status`, `route_status` |
| `/api/nodes/{id}/status` | GET | Detailed node state, active faults, and failover state machine |
| `/api/sensors` | GET | Latest readings across all nodes |
| `/api/sensors/{id}/history` | GET | Historical readings with time filtering |
| `/api/faults/active` | GET | Currently active unresolved system faults |
| `/api/anomalies` | GET | Recent statistical and ML anomaly events |

---

## 8. Automated Test Suite

Run the full automated test suite (59 passing tests):
```bash
python -m pytest tests/ -v -W ignore
```

### Test Suite Structure
* `tests/test_advancements.py` (8 tests): All 8 required advancement test scenarios:
  1. `test_01_normal_operation`: Dual direct routes, healthy network.
  2. `test_02_node1_failure`: Node 1 ESP32 timeout, FAULT 2 registered.
  3. `test_03_node1_sensor_failover`: Node 2 forwards Node 1 readings via CD4053; ESP32 OFFLINE, sensors AVAILABLE, route FAILOVER.
  4. `test_04_invalid_sensor_reading`: Invalid packets flagged, previous valid sensor readings preserved.
  5. `test_05_node1_recovery`: Valid direct packets confirm recovery, logged to `recovery_events`, route restores to DIRECT.
  6. `test_06_coordinator_failure`: Coordinator timeout triggers FAULT 5, nodes marked UNKNOWN.
  7. `test_07_duplicates_and_out_of_order`: Duplicate packets rejected, seq tracking preserved.
  8. `test_08_ml_anomaly_and_route_change_immunity`: ML flags genuine physical anomalies but remains immune to route switching.
* `tests/test_backend.py` (25 tests): Packet de-duplication, reboot detection, out-of-order handling, coordinator isolation.
* `tests/test_full_system.py` (20 tests): ML pipeline training/eval, edge metrics, REST API endpoints, processor lifecycle.
* `tests/test_fixes.py` (6 tests): Feature alignment, duplicate fault prevention, node status derivation.
