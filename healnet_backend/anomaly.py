"""
Sensor-behaviour anomaly detection for HEALNET.

This module ONLY judges whether *sensor readings* look unusual. It never decides
that a node or link has failed - that is the job of the deterministic
heartbeat/timeout logic in processor.py. An anomaly is therefore published as
"abnormal sensor behaviour" (FAULT 7) and never as an ESP32 hardware fault.

Three complementary detectors run on every sample:

1. Level   - robust z-score (median / MAD) against the node's own rolling window.
2. Rate    - change since the previous sample against a physical limit.
3. Pattern - IsolationForest on [temp, hum, light, d_temp, d_hum, d_light]
             (multivariate: catches combinations that are individually normal).

Then a cross-node check separates "the environment changed" (both nodes moved
the same way within a few seconds - e.g. someone switched the room light on)
from "this one sensor is misbehaving" (only one node moved).
"""
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

import config

log = logging.getLogger("healnet.anomaly")

try:
    from sklearn.ensemble import IsolationForest
    HAVE_SKLEARN = True
except Exception:                                   # scikit-learn is optional
    HAVE_SKLEARN = False

FEATURES = ("temp", "hum", "light")


@dataclass
class Finding:
    feature: str            # temp | hum | light | multivariate
    method: str             # mad | rate | iforest
    score: float            # 0..1+ (1 = twice the threshold)
    direction: int          # +1 / -1 / 0
    value: float
    detail: str
    classification: str = "pending"


@dataclass
class Result:
    anomalous: bool
    score: float
    findings: list


class NodeDetector:
    def __init__(self, node_id):
        self.node_id = node_id
        self.window = deque(maxlen=config.ML_WINDOW)      # (ts, temp, hum, light, flagged)
        self.prev = None                                  # (ts, temp, hum, light)
        self.last_z = {f: 0.0 for f in FEATURES}
        self.last_delta = {f: 0.0 for f in FEATURES}
        self.last_update_ts = 0.0
        self.last_multivar_ts = 0.0
        self.hist = deque(maxlen=12)                      # (ts, z, delta, multivariate_flag)
        self.model = None
        self.trained_on = 0
        self._since_train = 0

    # ---- warm start -------------------------------------------------------
    def seed(self, rows):
        """Pre-fill the window from stored readings after a backend restart."""
        for r in rows:
            if r.get("valid") and None not in (r.get("temp"), r.get("hum"), r.get("light")):
                self.window.append((r["ts"], r["temp"], r["hum"], r["light"], bool(r.get("anomaly"))))
                self.prev = (r["ts"], r["temp"], r["hum"], r["light"])

    # ---- main entry -------------------------------------------------------
    def update(self, ts, temp, hum, light):
        values = {"temp": float(temp), "hum": float(hum), "light": float(light)}
        findings = []
        max_abs_z = 0.0

        # 1) level: median / MAD over the node's own history
        if len(self.window) >= config.ML_MIN_SAMPLES:
            for i, f in enumerate(FEATURES):
                hist = np.array([w[i + 1] for w in self.window], dtype=float)
                med = float(np.median(hist))
                mad = float(np.median(np.abs(hist - med)))
                mad = max(mad, config.MAD_FLOOR[f])
                z = 0.6745 * (values[f] - med) / mad
                self.last_z[f] = z
                max_abs_z = max(max_abs_z, abs(z))
                if abs(z) > config.MAD_Z_THRESHOLD:
                    findings.append(Finding(
                        f, "mad", abs(z) / (2 * config.MAD_Z_THRESHOLD), 1 if z > 0 else -1, values[f],
                        f"{f}={values[f]:.1f} vs median {med:.1f} (robust z={z:+.1f})"))
        else:
            self.last_z = {f: 0.0 for f in FEATURES}

        # 2) rate of change since previous sample
        deltas = {f: 0.0 for f in FEATURES}
        if self.prev is not None:
            dt = max(0.5, ts - self.prev[0])
            if dt <= 4 * config.SENSOR_INTERVAL_S:        # do not compare across long outages
                scale = config.SENSOR_INTERVAL_S / dt
                for i, f in enumerate(FEATURES):
                    deltas[f] = (values[f] - self.prev[i + 1]) * scale
                    if abs(deltas[f]) > config.RATE_LIMITS[f]:
                        findings.append(Finding(
                            f, "rate", abs(deltas[f]) / (2 * config.RATE_LIMITS[f]),
                            1 if deltas[f] > 0 else -1, values[f],
                            f"{f} changed {deltas[f]:+.1f} in one sample (limit {config.RATE_LIMITS[f]})"))
        self.last_delta = deltas

        # 3) multivariate pattern (IsolationForest). Only counted when at least
        #    one other indicator is mildly elevated, to keep false alarms low.
        if self.model is not None:
            vec = np.array([[values["temp"], values["hum"], values["light"],
                             deltas["temp"], deltas["hum"], deltas["light"]]])
            if self.model.predict(vec)[0] == -1 and (findings or max_abs_z > 2.5):
                s = float(-self.model.decision_function(vec)[0])
                findings.append(Finding("multivariate", "iforest", min(1.0, 0.5 + s), 0, 0.0,
                                        f"IsolationForest outlier (score {s:.2f})"))
                self.last_multivar_ts = ts

        flagged = bool(findings)
        self.hist.append((ts, dict(self.last_z), dict(deltas), any(f.method == "iforest" for f in findings)))
        self.window.append((ts, values["temp"], values["hum"], values["light"], flagged))
        self.prev = (ts, values["temp"], values["hum"], values["light"])
        self.last_update_ts = ts
        self._since_train += 1
        self._maybe_train()

        score = min(1.0, max((f.score for f in findings), default=max_abs_z / (4 * config.MAD_Z_THRESHOLD)))
        return Result(flagged, round(score, 3), findings)

    # ---- IsolationForest training -----------------------------------------
    def _maybe_train(self):
        if not (config.ML_ENABLED and HAVE_SKLEARN):
            return
        n = len(self.window)
        due = (self.model is None and n >= config.ML_IFOREST_MIN_SAMPLES) or \
              (self.model is not None and self._since_train >= config.ML_IFOREST_RETRAIN_EVERY)
        if not due:
            return
        rows = [w for w in self.window if not w[4]]                    # train on clean samples only
        if len(rows) < config.ML_IFOREST_MIN_SAMPLES // 2:
            return
        X = []
        for a, b in zip(rows, rows[1:]):
            dt = max(0.5, b[0] - a[0])
            if dt > 4 * config.SENSOR_INTERVAL_S:
                continue
            k = config.SENSOR_INTERVAL_S / dt
            X.append([b[1], b[2], b[3], (b[1] - a[1]) * k, (b[2] - a[2]) * k, (b[3] - a[3]) * k])
        if len(X) < 40:
            return
        try:
            model = IsolationForest(n_estimators=100, contamination=config.ML_CONTAMINATION, random_state=42)
            model.fit(np.array(X))
            self.model, self.trained_on, self._since_train = model, len(X), 0
            log.info("node%d IsolationForest (re)trained on %d samples", self.node_id, len(X))
        except Exception as exc:
            log.warning("node%d IsolationForest training failed: %s", self.node_id, exc)

    # ---- cross-node support check ------------------------------------------
    def supports(self, finding, ts):
        """
        Did THIS node move the same way within +/- CORRELATION_WINDOW_S of the finding?
        True / False, or None when this node has no samples around that time (cannot compare).
        """
        near = [h for h in self.hist if abs(h[0] - ts) <= config.CORRELATION_WINDOW_S]
        if not near:
            return None
        f = finding.feature
        for _, z, delta, multivar in near:
            if f == "multivariate":
                if multivar:
                    return True
            elif finding.method == "rate":
                if delta[f] * finding.direction > 0.5 * config.RATE_LIMITS[f]:
                    return True
            elif z[f] * finding.direction >= config.CORRELATION_Z:
                return True
        return False


class AnomalyEngine:
    """Holds one detector per node and classifies findings across nodes."""

    def __init__(self, node_ids):
        self.detectors = {n: NodeDetector(n) for n in node_ids}

    def seed(self, node, rows):
        self.detectors[node].seed(rows)

    def process(self, node, ts, temp, hum, light):
        """Detect only. Classification is done later with classify() so the other node's
        sample for the same moment has time to arrive (nodes are not time-synchronised)."""
        return self.detectors[node].update(ts, temp, hum, light)

    def classify(self, node, finding, ts):
        others = [d for n, d in self.detectors.items() if n != node]
        verdicts = [d.supports(finding, ts) for d in others]
        known = [v for v in verdicts if v is not None]
        if not known:
            return "unconfirmed"            # no other node with fresh data to compare against
        if any(known):
            return "environmental_change"   # another node moved the same way -> real-world change
        return "localized_sensor_suspect"   # only this node moved -> suspect this sensor

    def summary(self):
        return {n: {"window": len(d.window), "iforest_trained_on": d.trained_on}
                for n, d in self.detectors.items()}
