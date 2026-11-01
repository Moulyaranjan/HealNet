# HEALNET dashboard (single file)

Live view of everything your backend publishes: node status, sensor values and charts,
network health, faults, anomalies, routing and (later) recovery.

## Run it
1. Replace your `mosquitto.conf` with the one in this folder (only change: a WebSocket
   listener on port 9001 added at the end). Restart Mosquitto: `mosquitto -c mosquitto.conf -v`
2. Start the backend as usual: `processor.py`, `api.py`, and `serial_bridge.py` (or `simulator.py --demo`).
3. Double-click `healnet_dashboard.html`. No install needed.

If the top-right chip says "Dashboard link: offline": the broker is not running, port 9001 is not
open, or the page could not download `mqtt.js` (needs internet once). For offline use, save
https://cdn.jsdelivr.net/npm/mqtt@5/dist/mqtt.min.js next to the HTML file as `mqtt.min.js`.

Broker and API addresses can be changed at the top of the page (e.g. when the dashboard runs on
another computer, use `ws://<laptop-ip>:9001` and `http://<laptop-ip>:8000`; start `api.py --host 0.0.0.0`).

## Where each panel gets its data
| Panel | Source |
|---|---|
| Overview tiles, health %, active faults | `healnet/system/summary` (retained) |
| Node cards, values, charts | `healnet/nodeN/status` (retained), `healnet/nodeN/processed`; history from `/api/sensors/N/history` |
| Fault & event log | `healnet/system/fault`; backfill `/api/faults/history` |
| Anomalies | `healnet/system/anomaly`; backfill `/api/anomalies` |
| Backend / bridge / coordinator chips | `healnet/system/backend`, `/bridge`, `/coordinator` (retained, Last-Will) |
| Routing | `route` field of node status + `summary.coordinator` |
| Recovery | `healnet/system/recovery` (empty until the recovery phase exists) |

Not working yet, by design: recovery buttons (disabled), real multi-hop routing display (the backend only reports what the firmware reports).
