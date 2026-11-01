"""
HEALNET backend configuration.

Every value can be overridden with an environment variable (HEALNET_<NAME>),
so the same code runs unchanged on a laptop, a Raspberry Pi, or in Docker.
"""
import os


def _env(name, default, cast=str):
    raw = os.environ.get(f"HEALNET_{name}")
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


# --------------------------------------------------------------------------
# MQTT broker (local Mosquitto)
# --------------------------------------------------------------------------
MQTT_HOST = _env("MQTT_HOST", "127.0.0.1")
MQTT_PORT = _env("MQTT_PORT", 1883, int)
MQTT_USER = _env("MQTT_USER", "")
MQTT_PASS = _env("MQTT_PASS", "")
MQTT_KEEPALIVE = _env("MQTT_KEEPALIVE", 30, int)

# --------------------------------------------------------------------------
# Network layout
# --------------------------------------------------------------------------
TOPIC_ROOT = "healnet"
NODE_IDS = (1, 2)                       # sensor nodes (coordinator is not a sensor node)

# --------------------------------------------------------------------------
# Serial bridge (coordinator ESP32 -> USB -> laptop)
# --------------------------------------------------------------------------
SERIAL_PORT = _env("SERIAL_PORT", "")   # "" = auto-detect (CP210x / CH340 / FTDI)
SERIAL_BAUD = _env("SERIAL_BAUD", 115200, int)
COORDINATOR_TIMEOUT_S = _env("COORDINATOR_TIMEOUT_S", 6.0, float)   # no serial line for this long => coordinator offline
COORDINATOR_PUBLISH_S = _env("COORDINATOR_PUBLISH_S", 2.0, float)
SPOOL_DIR = _env("SPOOL_DIR", os.path.join(os.path.dirname(__file__), "spool"))
SPOOL_MAX_LINES = _env("SPOOL_MAX_LINES", 50000, int)

# --------------------------------------------------------------------------
# Timing (must stay consistent with the firmware)
# --------------------------------------------------------------------------
SENSOR_INTERVAL_S = _env("SENSOR_INTERVAL_S", 2.0, float)   # nodes transmit every 2 s
NODE_STALE_S = _env("NODE_STALE_S", 6.0, float)             # ~3 missed packets   -> STALE
NODE_TIMEOUT_S = _env("NODE_TIMEOUT_S", 10.0, float)        # same as coordinator -> OFFLINE
STARTUP_GRACE_S = _env("STARTUP_GRACE_S", 12.0, float)      # no timeout faults right after backend start
DELAYED_PACKET_S = _env("DELAYED_PACKET_S", 6.0, float)     # older than this on arrival => "delayed"/backfill
OOO_WINDOW = _env("OOO_WINDOW", 10, int)                    # seq this far behind = out-of-order, further = reboot
DEDUP_WINDOW = _env("DEDUP_WINDOW", 256, int)               # remembered sequence numbers per node
GAP_EVENT_MIN = _env("GAP_EVENT_MIN", 3, int)               # publish a packet-loss event when >= this many missing

# --------------------------------------------------------------------------
# Plausible sensor ranges (DHT11: 0-50 C, 20-90 %RH; ESP32 ADC: 12-bit)
# --------------------------------------------------------------------------
TEMP_RANGE = (-5.0, 60.0)
HUM_RANGE = (0.0, 100.0)
LIGHT_RANGE = (0, 4095)
LDR_RAIL_LOW = _env("LDR_RAIL_LOW", 3, int)                 # <= this counts as "stuck low"
LDR_RAIL_HIGH = _env("LDR_RAIL_HIGH", 4092, int)            # >= this counts as "stuck high"
SENSOR_INVALID_STREAK = _env("SENSOR_INVALID_STREAK", 3, int)   # consecutive bad packets before FAULT 1
LDR_RAIL_STREAK = _env("LDR_RAIL_STREAK", 15, int)              # consecutive rail readings before "suspected"
SENSOR_RECOVER_STREAK = _env("SENSOR_RECOVER_STREAK", 3, int)   # consecutive good packets to clear FAULT 1

# --------------------------------------------------------------------------
# Machine learning / statistics
# --------------------------------------------------------------------------
ML_ENABLED = _env("ML_ENABLED", True, bool)
ML_WINDOW = _env("ML_WINDOW", 300, int)                     # samples kept per node (~10 min at 2 s)
ML_MIN_SAMPLES = _env("ML_MIN_SAMPLES", 30, int)            # warm-up before statistical detection
ML_IFOREST_MIN_SAMPLES = _env("ML_IFOREST_MIN_SAMPLES", 120, int)
ML_IFOREST_RETRAIN_EVERY = _env("ML_IFOREST_RETRAIN_EVERY", 150, int)
ML_CONTAMINATION = _env("ML_CONTAMINATION", 0.02, float)
MAD_Z_THRESHOLD = _env("MAD_Z_THRESHOLD", 4.0, float)       # modified z-score
# Max believable change between two consecutive samples (2 s apart)
RATE_LIMITS = {"temp": 4.0, "hum": 12.0, "light": 1800.0}
# Floors for MAD so a flat signal (DHT11 has 1-degree resolution!) cannot give z = infinity
MAD_FLOOR = {"temp": 0.5, "hum": 2.0, "light": 40.0}
CORRELATION_WINDOW_S = _env("CORRELATION_WINDOW_S", 8.0, float)
CORRELATION_Z = _env("CORRELATION_Z", 2.0, float)
ANOMALY_PERSIST_COUNT = _env("ANOMALY_PERSIST_COUNT", 5, int)   # anomalous samples in last 10 => FAULT 7
ANOMALY_CLEAR_COUNT = _env("ANOMALY_CLEAR_COUNT", 10, int)      # clean samples in a row => clear FAULT 7

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
DB_PATH = _env("DB_PATH", os.path.join(os.path.dirname(__file__), "data", "healnet.db"))
RETENTION_DAYS = _env("RETENTION_DAYS", 30, int)
SUMMARY_PUBLISH_S = _env("SUMMARY_PUBLISH_S", 2.0, float)
STATUS_REFRESH_S = _env("STATUS_REFRESH_S", 5.0, float)

# --------------------------------------------------------------------------
# Fault catalogue (matches the classification in the HEALNET design)
# --------------------------------------------------------------------------
FAULTS = {
    1: ("SENSOR_FAILURE", "MEDIUM"),
    2: ("NODE_COMM_FAILURE", "HIGH"),          # node silent, cause (link vs power) not yet known
    3: ("NODE_POWER_FAILURE", "HIGH"),         # RESERVED - needs relay/current feedback
    4: ("WIRELESS_LINK_FAILURE", "HIGH"),      # RESERVED - needs RSSI/ack/multi-hop information
    5: ("COORDINATOR_FAILURE", "CRITICAL"),
    6: ("MQTT_BACKEND_FAILURE", "CRITICAL"),
    7: ("ABNORMAL_SENSOR_BEHAVIOUR", "LOW"),
}

# --------------------------------------------------------------------------
# Topics
# --------------------------------------------------------------------------
def t_data(n):        return f"{TOPIC_ROOT}/node{n}/data"          # raw envelope from the bridge
def t_processed(n):   return f"{TOPIC_ROOT}/node{n}/processed"     # validated + enriched by backend
def t_status(n):      return f"{TOPIC_ROOT}/node{n}/status"        # retained node health
T_FAULT = f"{TOPIC_ROOT}/system/fault"
T_RECOVERY = f"{TOPIC_ROOT}/system/recovery"                       # RESERVED (recovery is a later phase)
T_ROUTING = f"{TOPIC_ROOT}/system/routing"
T_ANOMALY = f"{TOPIC_ROOT}/system/anomaly"
T_SUMMARY = f"{TOPIC_ROOT}/system/summary"                         # retained network overview
T_COORDINATOR = f"{TOPIC_ROOT}/system/coordinator"                 # retained, written by the bridge
T_COORD_EVENT = f"{TOPIC_ROOT}/system/coordinator_event"
T_BACKEND = f"{TOPIC_ROOT}/system/backend"                         # retained, LWT of the processor
T_BRIDGE = f"{TOPIC_ROOT}/system/bridge"                           # retained, LWT of the serial bridge
T_STATS = f"{TOPIC_ROOT}/system/stats"
