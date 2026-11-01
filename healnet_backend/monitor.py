"""
Live terminal view of everything HEALNET publishes (a poor man's dashboard).

    python monitor.py              # everything under healnet/#
    python monitor.py --only fault anomaly status
"""
import argparse
import json
import time

import paho.mqtt.client as mqtt

import config

COLORS = {"fault": "\033[91m", "anomaly": "\033[95m", "status": "\033[96m", "routing": "\033[94m",
          "summary": "\033[93m", "processed": "\033[92m", "data": "\033[90m", "coordinator": "\033[96m",
          "backend": "\033[93m", "bridge": "\033[93m"}
RESET = "\033[0m"


def kind(topic):
    last = topic.split("/")[-1]
    return last if last in COLORS else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="show only these kinds (fault anomaly status routing summary "
                                              "processed data coordinator backend bridge)")
    args = ap.parse_args()

    def on_connect(client, userdata, flags, rc, props):
        print(f"connected to {config.MQTT_HOST}:{config.MQTT_PORT} - listening on {config.TOPIC_ROOT}/#")
        client.subscribe(f"{config.TOPIC_ROOT}/#", qos=0)

    def on_message(client, userdata, msg):
        k = kind(msg.topic)
        if args.only and k not in args.only:
            return
        try:
            body = json.dumps(json.loads(msg.payload), separators=(",", ":"))
        except ValueError:
            body = msg.payload.decode("utf-8", "replace")
        retained = " (retained)" if msg.retain else ""
        print(f"{COLORS.get(k, '')}{time.strftime('%H:%M:%S')} {msg.topic}{retained}\n    {body[:300]}{RESET}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="healnet-monitor", protocol=mqtt.MQTTv311)
    if config.MQTT_USER:
        client.username_pw_set(config.MQTT_USER, config.MQTT_PASS)
    client.on_connect, client.on_message = on_connect, on_message
    client.connect(config.MQTT_HOST, config.MQTT_PORT, config.MQTT_KEEPALIVE)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
