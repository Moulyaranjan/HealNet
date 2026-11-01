"""Shared MQTT client construction (paho-mqtt >= 2.0)."""
import json
import logging

import paho.mqtt.client as mqtt

import config

log = logging.getLogger("healnet.mqtt")


def make_client(client_id, lwt_topic=None, lwt_payload=None, persistent=False):
    """
    persistent=True  -> clean_session=False with a fixed client_id. While this
    client is down the broker queues its QoS 1 subscriptions and delivers them
    on reconnect (this is how the backend survives a restart without losing data).
    """
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=mqtt.MQTTv311,
        clean_session=not persistent,
    )
    if config.MQTT_USER:
        client.username_pw_set(config.MQTT_USER, config.MQTT_PASS)
    if lwt_topic:
        client.will_set(lwt_topic, json.dumps(lwt_payload or {"status": "offline"}), qos=1, retain=True)
    client.reconnect_delay_set(min_delay=1, max_delay=15)
    client.max_queued_messages_set(0)          # 0 = unlimited in-memory queue while reconnecting
    client.max_inflight_messages_set(100)
    return client


def connect_async(client):
    """Non-blocking connect: loop_start() keeps retrying until the broker is up."""
    client.connect_async(config.MQTT_HOST, config.MQTT_PORT, config.MQTT_KEEPALIVE)
    client.loop_start()


def publish_json(client, topic, payload, qos=0, retain=False):
    try:
        info = client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=qos, retain=retain)
        return info.rc == mqtt.MQTT_ERR_SUCCESS
    except Exception as exc:                    # never let a publish kill a worker thread
        log.warning("publish to %s failed: %s", topic, exc)
        return False
