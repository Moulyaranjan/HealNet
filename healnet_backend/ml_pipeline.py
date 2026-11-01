"""
ml_pipeline.py — HEALNET ML Training, Evaluation, and Persistence Module
=========================================================================

Responsibilities
----------------
* Generate synthetic ESP32/DHT11/LDR sensor data (normal + abnormal) when no
  real database readings are available.
* Engineer features: rolling mean, rolling std, rate of change, inter-node diff.
* Train and compare IsolationForest, LocalOutlierFactor, and One-Class SVM.
* Time-aware train/test split (first 70% → train, last 30% → test).
* Evaluate with precision, recall, F1, accuracy, confusion matrix, FPR, FNR.
* Persist the best model to  models/anomaly_model.joblib  and metadata to
  models/anomaly_metadata.json.
* Expose a standalone  predict(features_dict)  function and a  ModelManager
  class for use by the API and processor.

Imports from existing modules: config, storage, anomaly (read-only).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Existing HEALNET modules (imported read-only – never modified)
# ---------------------------------------------------------------------------
import config  # noqa: F401  – used for sensor range constants
from anomaly import FEATURES  # ("temp", "hum", "light")
from storage import Storage  # used in ModelManager.retrain()

# ---------------------------------------------------------------------------
# Optional sklearn imports – module is still importable without sklearn but
# training / prediction will raise an informative RuntimeError.
# ---------------------------------------------------------------------------
try:
    import joblib
    from sklearn.covariance import EllipticEnvelope  # noqa: F401 – kept as fallback reference
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False

log = logging.getLogger("healnet.ml_pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_MODULE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "anomaly_metadata.json")
EVAL_REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")

# ---------------------------------------------------------------------------
# Feature column names (after engineering)
# ---------------------------------------------------------------------------
BASE_FEATURES = ["temp", "hum", "light"]
ENGINEERED_FEATURES = [
    "temp", "hum", "light",
    "temp_roll_mean", "hum_roll_mean", "light_roll_mean",
    "temp_roll_std",  "hum_roll_std",  "light_roll_std",
    "temp_roc",       "hum_roc",       "light_roc",
    "temp_diff_node", "hum_diff_node", "light_diff_node",
]

ROLLING_WINDOW = 10  # samples (matches requirement §3)

# ============================================================================
#  SECTION 1 – Synthetic data generation
# ============================================================================

def _generate_normal_samples(
    n: int,
    node_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate *n* normal ESP32/DHT11/LDR readings for *node_id*.

    Sensor characteristics
    ----------------------
    * DHT11 – 0–50 °C, 20–90 %RH, 1 °C / 1 % resolution → Gaussian
      centred on comfortable indoor conditions with small step-noise.
    * LDR   – 0–4095 ADC counts; daytime ~1500–3000, night-time ~50–300,
      simulated as a slow sinusoid plus white noise.

    Returns
    -------
    ndarray of shape (n, 3) – columns: temp, hum, light.
    """
    # Temperature: mean ~24 °C, SD 2 °C, quantised to 1 °C (DHT11 resolution)
    temp_raw = rng.normal(24.0 + node_id * 0.5, 2.0, n)
    temp = np.clip(np.round(temp_raw), 0, 50).astype(float)

    # Humidity: mean ~55 %, SD 5 %, quantised to 1 % (DHT11 resolution)
    hum_raw = rng.normal(55.0 - node_id * 2.0, 5.0, n)
    hum = np.clip(np.round(hum_raw), 20, 90).astype(float)

    # Light: slow day/night sinusoid (period = n/2) + Gaussian noise
    phase = 2 * np.pi * np.arange(n) / max(n // 2, 1)
    light_base = 1800 + 1200 * np.sin(phase + node_id * 0.3)
    light_noise = rng.normal(0, 120, n)
    light = np.clip(np.round(light_base + light_noise), 0, 4095).astype(float)

    return np.column_stack([temp, hum, light])


def _generate_abnormal_samples(
    n: int,
    node_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate *n* *abnormal* readings representing sensor faults and
    environmental anomalies.

    Fault types injected (chosen uniformly at random per sample)
    -----------------------------------------------------------
    1. Stuck sensor  – temp or hum freezes at a constant value.
    2. Sensor spike  – sudden extreme value (near the sensor rail).
    3. Impossible combination – e.g. 5 °C + 95 % RH (condensation range,
       beyond DHT11 spec) or 45 °C + 10 % RH simultaneously.
    4. LDR rail glitch – light ≤ 3 or ≥ 4092 (stuck ADC pin).
    5. Fast oscillation – alternating between two extreme values.

    Returns
    -------
    ndarray of shape (n, 3) – columns: temp, hum, light.
    """
    temp_arr  = np.zeros(n)
    hum_arr   = np.zeros(n)
    light_arr = np.zeros(n)

    fault_types = rng.integers(0, 5, n)  # 0–4 inclusive

    for i in range(n):
        ft = fault_types[i]

        if ft == 0:
            # --- Fault 0: stuck sensor (constant temperature or humidity) ---
            temp_arr[i]  = rng.choice([0.0, 50.0])   # rail value
            hum_arr[i]   = rng.choice([20.0, 90.0])
            light_arr[i] = float(rng.integers(100, 3000))

        elif ft == 1:
            # --- Fault 1: extreme spike ---
            temp_arr[i]  = rng.choice([-4.0, 58.0])
            hum_arr[i]   = rng.choice([1.0, 99.0])
            light_arr[i] = float(rng.integers(0, 4096))

        elif ft == 2:
            # --- Fault 2: impossible combination ---
            # Pick a physically contradictory (temp, hum) pair
            if rng.random() > 0.5:
                temp_arr[i] = float(rng.integers(40, 51))   # very hot
                hum_arr[i]  = float(rng.integers(90, 101))  # very humid
            else:
                temp_arr[i] = float(rng.integers(0, 6))     # very cold
                hum_arr[i]  = float(rng.integers(0, 11))    # extremely dry
            light_arr[i] = float(rng.integers(0, 4096))

        elif ft == 3:
            # --- Fault 3: LDR ADC rail glitch ---
            temp_arr[i]  = rng.normal(24.0, 2.0)
            hum_arr[i]   = rng.normal(55.0, 5.0)
            light_arr[i] = float(rng.choice([0, 1, 2, 4093, 4094, 4095]))

        else:
            # --- Fault 4: fast oscillation (extreme alternation) ---
            if i % 2 == 0:
                temp_arr[i]  = float(rng.integers(45, 51))
                hum_arr[i]   = float(rng.integers(85, 91))
                light_arr[i] = 4095.0
            else:
                temp_arr[i]  = float(rng.integers(0, 6))
                hum_arr[i]   = float(rng.integers(20, 26))
                light_arr[i] = 0.0

    # Clip to plausible physical bounds so sklearn still sees "sensor-like" data
    temp_arr  = np.clip(temp_arr,  *config.TEMP_RANGE)
    hum_arr   = np.clip(hum_arr,   *config.HUM_RANGE)
    light_arr = np.clip(light_arr, *config.LIGHT_RANGE)

    return np.column_stack([temp_arr, hum_arr, light_arr])


def generate_synthetic_dataset(
    n_normal: int = 1000,
    n_abnormal: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a realistic labelled synthetic dataset for two virtual nodes.

    Normal readings follow continuous physical dynamics (diurnal cycles + sensor noise).
    Abnormal readings (faults/anomalies) are injected into episodes distributed throughout
    the timeline, ensuring realistic rolling statistics and no train/test data leakage.
    """
    rng = np.random.default_rng(seed)

    log.info(
        "Generating synthetic dataset: %d normal + %d abnormal samples distributed over timeline",
        n_normal, n_abnormal,
    )

    n_total = n_normal + n_abnormal
    n1_total = n_total // 2
    n2_total = n_total - n1_total

    n1_ab = n_abnormal // 2
    n2_ab = n_abnormal - n1_ab

    # Generate continuous baseline time series for both nodes
    X1 = _generate_normal_samples(n1_total, node_id=1, rng=rng)
    y1 = np.zeros(n1_total, dtype=int)
    if n1_ab > 0:
        idx1 = rng.choice(n1_total, size=n1_ab, replace=False)
        X1[idx1] = _generate_abnormal_samples(n1_ab, node_id=1, rng=rng)
        y1[idx1] = 1

    X2 = _generate_normal_samples(n2_total, node_id=2, rng=rng)
    y2 = np.zeros(n2_total, dtype=int)
    if n2_ab > 0:
        idx2 = rng.choice(n2_total, size=n2_ab, replace=False)
        X2[idx2] = _generate_abnormal_samples(n2_ab, node_id=2, rng=rng)
        y2[idx2] = 1

    # Interleave node 1 and node 2 readings to simulate alternating network traffic
    X = np.empty((n_total, 3), dtype=float)
    y = np.empty(n_total, dtype=int)
    X[0::2] = X1[:len(X[0::2])]
    X[1::2] = X2[:len(X[1::2])]
    y[0::2] = y1[:len(y[0::2])]
    y[1::2] = y2[:len(y[1::2])]

    log.debug("Synthetic X shape: %s, y shape: %s (realistic timeline)", X.shape, y.shape)
    return X, y


# ============================================================================
#  SECTION 2 – Feature engineering
# ============================================================================

def engineer_features(
    X_raw: np.ndarray,
    window: int = ROLLING_WINDOW,
) -> np.ndarray:
    """
    Derive engineered features from raw (temp, hum, light) columns.

    Features added
    --------------
    * rolling_mean_{temp,hum,light}  – window-sample trailing mean.
    * rolling_std_{temp,hum,light}   – window-sample trailing std (Bessel-corrected).
    * roc_{temp,hum,light}           – rate of change (diff vs. previous sample).
    * diff_node_{temp,hum,light}     – difference from the column-wise global mean
                                       (proxy for inter-node divergence; exact per-node
                                       diff requires aligned timestamps, so we use the
                                       batch mean as a reference when node labels are
                                       unavailable at feature-engineering time).

    Parameters
    ----------
    X_raw  : ndarray (N, 3) – columns: temp, hum, light (in that order).
    window : rolling window size.

    Returns
    -------
    ndarray of shape (N, 15) – one row per input sample.
    """
    N = X_raw.shape[0]

    # Pre-allocate output
    out = np.zeros((N, len(ENGINEERED_FEATURES)), dtype=float)

    # Base columns
    out[:, 0] = X_raw[:, 0]  # temp
    out[:, 1] = X_raw[:, 1]  # hum
    out[:, 2] = X_raw[:, 2]  # light

    for col_idx, col_name in enumerate(BASE_FEATURES):
        col = X_raw[:, col_idx].astype(float)

        # ---- rolling mean and std ----------------------------------------
        roll_mean = np.full(N, np.nan)
        roll_std  = np.full(N, np.nan)
        for i in range(N):
            start = max(0, i - window + 1)
            segment = col[start : i + 1]
            roll_mean[i] = np.mean(segment)
            roll_std[i]  = float(np.std(segment, ddof=min(1, len(segment) - 1)))

        # Fill leading NaN with first valid value
        roll_mean = np.where(np.isnan(roll_mean), col, roll_mean)
        roll_std  = np.where(np.isnan(roll_std),  0.0, roll_std)

        # ---- rate of change (diff vs. previous) --------------------------
        roc = np.zeros(N)
        roc[1:] = col[1:] - col[:-1]
        # First sample has no predecessor → 0

        # ---- inter-node diff (causal EMA proxy, no data leakage) ----------
        # Using an exponential moving average as a causal reference avoids
        # the data-leakage of subtracting the global batch mean, which would
        # let test-set values influence training features.
        alpha = 0.1
        ema = np.zeros(N)
        ema[0] = col[0]
        for _ema_i in range(1, N):
            ema[_ema_i] = alpha * col[_ema_i] + (1.0 - alpha) * ema[_ema_i - 1]
        diff_node = col - ema

        # Store: 3 base + 3 roll_mean + 3 roll_std + 3 roc + 3 diff_node
        out[:, 3 + col_idx]  = roll_mean
        out[:, 6 + col_idx]  = roll_std
        out[:, 9 + col_idx]  = roc
        out[:, 12 + col_idx] = diff_node

    return out


# ============================================================================
#  SECTION 3 – Train / test split (time-aware)
# ============================================================================

def time_aware_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset preserving temporal order (first 70 % → train, last 30 % → test).

    Parameters
    ----------
    X          : feature matrix (N, F).
    y          : label vector (N,) — 0=normal, 1=abnormal.
    train_frac : fraction of samples going to the training set.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    split = int(len(X) * train_frac)
    log.info(
        "Time-aware split: %d train / %d test  (%.0f%% / %.0f%%)",
        split, len(X) - split, train_frac * 100, (1 - train_frac) * 100,
    )
    return X[:split], X[split:], y[:split], y[split:]


# ============================================================================
#  SECTION 4 – Model training pipeline
# ============================================================================

def _sklearn_required() -> None:
    if not HAVE_SKLEARN:
        raise RuntimeError(
            "scikit-learn is not installed. "
            "Install it with:  pip install scikit-learn"
        )


def _contamination_from_labels(y_train: np.ndarray) -> float:
    """Estimate contamination from the fraction of abnormal labels in y_train."""
    frac = float(np.mean(y_train == 1))
    # Clamp to the range sklearn accepts; default to config value if training set is clean
    return float(np.clip(frac if frac > 0 else config.ML_CONTAMINATION, 0.01, 0.49))


def _build_models(contamination: float) -> dict[str, Any]:
    """Instantiate the three candidate outlier detectors."""
    return {
        "IsolationForest": IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        ),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,        # needed so we can call predict() on new data
        ),
        "OneClassSVM": OneClassSVM(
            kernel="rbf",
            nu=contamination,
            gamma="scale",
        ),
    }


# ============================================================================
#  SECTION 5 – Evaluation
# ============================================================================

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """
    Evaluate a fitted novelty-detection model on labelled test data.

    Sklearn outlier detectors return +1 (inlier/normal) and -1 (outlier/abnormal).
    We remap: −1 → 1 (abnormal), +1 → 0 (normal) to match our label convention.

    All metric values come from actual sklearn calls — never fabricated.

    Returns
    -------
    dict with keys: model_name, precision, recall, f1, accuracy,
                    confusion_matrix, fpr, fnr, raw_predictions.
    """
    raw_preds = model.predict(X_test)           # +1 / -1
    y_pred    = np.where(raw_preds == -1, 1, 0) # remap to 0/1

    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    acc  = float(accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)       # shape (2, 2): [[TN,FP],[FN,TP]]
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0   # False Positive Rate
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0   # False Negative Rate

    log.info(
        "[%s] precision=%.3f recall=%.3f f1=%.3f acc=%.3f fpr=%.3f fnr=%.3f",
        model_name, prec, rec, f1, acc, fpr, fnr,
    )

    return {
        "model_name":       model_name,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "accuracy":         acc,
        "confusion_matrix": cm.tolist(),
        "fpr":              fpr,
        "fnr":              fnr,
        "raw_predictions":  y_pred,
    }


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted side-by-side comparison of model evaluation metrics."""
    header = f"{'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'FPR':>8} {'FNR':>8}"
    sep    = "-" * len(header)
    print()
    print("=" * len(header))
    print("  HEALNET ML Pipeline — Model Comparison")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['model_name']:<22}"
            f"{r['precision']:>10.4f}"
            f"{r['recall']:>8.4f}"
            f"{r['f1']:>8.4f}"
            f"{r['accuracy']:>10.4f}"
            f"{r['fpr']:>8.4f}"
            f"{r['fnr']:>8.4f}"
        )
    print(sep)
    best = max(results, key=lambda r: r["f1"])
    print(f"  Best model (by F1): {best['model_name']}  (F1={best['f1']:.4f})")
    print("=" * len(header))
    print()


# ============================================================================
#  SECTION 6 – Model persistence
# ============================================================================

def _ensure_models_dir() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(
    model: Any,
    scaler: StandardScaler,
    best_result: dict[str, Any],
    *,
    synthetic: bool,
    training_samples: int,
    all_results: list[dict[str, Any]] | None = None,
    test_samples: int | None = None,
) -> None:
    """
    Persist the best model artefact and its metadata JSON.

    Parameters
    ----------
    model            : fitted sklearn estimator (best one).
    scaler           : fitted StandardScaler used before the model.
    best_result      : evaluation metrics dict from evaluate_model().
    synthetic        : True when trained on generated data (required by §10).
    training_samples : number of samples used for training.
    all_results      : list of evaluation dicts for all compared models.
    test_samples     : number of test samples.
    """
    _ensure_models_dir()

    payload = {"model": model, "scaler": scaler}
    joblib.dump(payload, MODEL_PATH)
    log.info("Model saved -> %s", MODEL_PATH)

    now_iso = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_type":       best_result["model_name"],
        "trained_at":       now_iso,
        "training_samples": training_samples,
        "synthetic":        synthetic,          # §10 – clearly marked
        "features":         ENGINEERED_FEATURES,
        "rolling_window":   ROLLING_WINDOW,
        "evaluation": {
            "precision":        best_result["precision"],
            "recall":           best_result["recall"],
            "f1":               best_result["f1"],
            "accuracy":         best_result["accuracy"],
            "fpr":              best_result["fpr"],
            "fnr":              best_result["fnr"],
            "confusion_matrix": best_result["confusion_matrix"],
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("Metadata saved -> %s", METADATA_PATH)

    # Save comprehensive evaluation report
    report_models = []
    candidates = all_results if all_results else [best_result]
    for r in candidates:
        report_models.append({
            "model_name":       r["model_name"],
            "precision":        r["precision"],
            "recall":           r["recall"],
            "f1":               r["f1"],
            "accuracy":         r["accuracy"],
            "fpr":              r["fpr"],
            "fnr":              r["fnr"],
            "confusion_matrix": r["confusion_matrix"],
        })

    eval_report = {
        "timestamp":        now_iso,
        "synthetic":        synthetic,
        "training_samples": training_samples,
        "test_samples":     test_samples or 0,
        "best_model":       best_result["model_name"],
        "best_f1":          best_result["f1"],
        "models_evaluated": report_models,
    }
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(eval_report, fh, indent=2)
    log.info("Evaluation report saved -> %s", EVAL_REPORT_PATH)


def load_model() -> tuple[Any, StandardScaler] | tuple[None, None]:
    """
    Load the persisted model and scaler from disk.

    Returns
    -------
    (model, scaler) if found, (None, None) otherwise.
    """
    if not os.path.isfile(MODEL_PATH):
        log.warning("No saved model found at %s", MODEL_PATH)
        return None, None
    payload = joblib.load(MODEL_PATH)
    log.info("Model loaded from %s", MODEL_PATH)
    return payload["model"], payload["scaler"]


def load_metadata() -> dict[str, Any]:
    """Load model metadata JSON. Returns empty dict if file is absent."""
    if not os.path.isfile(METADATA_PATH):
        return {}
    with open(METADATA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ============================================================================
#  SECTION 7 – Full training pipeline (orchestrator)
# ============================================================================

def run_training_pipeline(
    X_raw: np.ndarray | None = None,
    y_raw: np.ndarray | None = None,
    *,
    synthetic: bool | None = None,
    n_normal:   int = 1500,
    n_abnormal: int = 150,
) -> dict[str, Any]:
    """
    End-to-end training pipeline.

    1. Use provided (X_raw, y_raw) or generate a synthetic dataset.
    2. Engineer features.
    3. Time-aware split.
    4. Scale features.
    5. Train IsolationForest, LocalOutlierFactor, OneClassSVM.
    6. Evaluate each on the test split.
    7. Print comparison table.
    8. Save the best model (by F1).

    Returns
    -------
    dict with 'best_model', 'best_result', 'all_results', 'scaler', 'synthetic'.
    """
    _sklearn_required()

    # ----- Step 1: data source --------------------------------------------
    if X_raw is None or y_raw is None:
        log.info("No real data provided — using synthetic dataset.")
        X_raw, y_raw = generate_synthetic_dataset(n_normal=n_normal, n_abnormal=n_abnormal)
        synthetic = True
    else:
        if synthetic is None:
            synthetic = False
        log.info("Training on provided dataset: %d samples", len(X_raw))

    # ----- Step 2: feature engineering ------------------------------------
    log.info("Engineering features (window=%d) ...", ROLLING_WINDOW)
    X_eng = engineer_features(X_raw, window=ROLLING_WINDOW)

    # ----- Step 3: time-aware split ---------------------------------------
    X_train, X_test, y_train, y_test = time_aware_split(X_eng, y_raw)

    # ----- Step 4: feature scaling ----------------------------------------
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ----- Step 5: train models -------------------------------------------
    contamination = _contamination_from_labels(y_train)
    log.info("Estimated contamination for training: %.4f", contamination)

    models = _build_models(contamination)

    fitted: dict[str, Any] = {}
    for name, mdl in models.items():
        log.info("Training %s ...", name)
        try:
            # One-class estimators are trained on inliers only
            X_inliers = X_train_sc[y_train == 0]
            if len(X_inliers) < 10:
                log.warning("%s: too few inlier samples (%d) — skipping.", name, len(X_inliers))
                continue
            mdl.fit(X_inliers)
            fitted[name] = mdl
            log.info("%s training complete.", name)
        except Exception as exc:
            log.error("Failed to train %s: %s", name, exc)

    if not fitted:
        raise RuntimeError("No models trained successfully.")

    # ----- Step 6: evaluate -----------------------------------------------
    all_results: list[dict[str, Any]] = []
    for name, mdl in fitted.items():
        result = evaluate_model(mdl, X_test_sc, y_test, name)
        all_results.append(result)

    # ----- Step 7: print comparison table ---------------------------------
    print_comparison_table(all_results)

    # ----- Step 8: save best model ----------------------------------------
    best_result = max(all_results, key=lambda r: r["f1"])
    best_model  = fitted[best_result["model_name"]]

    save_model(
        best_model,
        scaler,
        best_result,
        synthetic=bool(synthetic),
        training_samples=len(X_train),
        all_results=all_results,
        test_samples=len(X_test),
    )

    return {
        "best_model":  best_model,
        "best_result": best_result,
        "all_results": all_results,
        "scaler":      scaler,
        "synthetic":   synthetic,
    }


# ============================================================================
#  SECTION 8 – Standalone predict() function
# ============================================================================

def predict(features_dict: dict[str, float]) -> tuple[bool, float]:
    """
    Load the saved model from disk and score a single sample.

    Parameters
    ----------
    features_dict : dict with at minimum keys ``temp``, ``hum``, ``light``.
                    Optional engineered features are accepted if present.

    Returns
    -------
    (anomalous: bool, score: float)
        score is in [0, 1]: 0.0 = normal centre, 1.0 = strong outlier.
    """
    _sklearn_required()

    model, scaler = load_model()
    if model is None or scaler is None:
        raise RuntimeError(
            "No saved model found. Run run_training_pipeline() first."
        )

    # Build the raw feature vector from the dict
    temp  = float(features_dict.get("temp",  24.0))
    hum   = float(features_dict.get("hum",   55.0))
    light = float(features_dict.get("light", 1800.0))

    raw = np.array([[temp, hum, light]])
    eng = engineer_features(raw, window=ROLLING_WINDOW)  # shape (1, 15)
    eng_sc = scaler.transform(eng)

    raw_pred = model.predict(eng_sc)[0]              # +1 (normal) or -1 (outlier)
    anomalous = bool(raw_pred == -1)

    # Normalise decision score to [0, 1]
    try:
        df = float(model.decision_function(eng_sc)[0])
        # decision_function: positive = inlier, negative = outlier
        # Map: df → 0 when inlier (high df), → 1 when strong outlier (very negative df)
        score = float(np.clip(0.5 - df / 2.0, 0.0, 1.0))
    except AttributeError:
        # LocalOutlierFactor with novelty=True exposes decision_function,
        # but guard against edge-cases.
        score = 1.0 if anomalous else 0.0

    log.debug(
        "predict: temp=%.1f hum=%.1f light=%.0f → anomalous=%s score=%.3f",
        temp, hum, light, anomalous, score,
    )
    return anomalous, round(score, 4)


# ============================================================================
#  SECTION 9 – ModelManager class
# ============================================================================

class ModelManager:
    """
    High-level interface for the saved anomaly-detection model.

    Designed for import by the HEALNET API and processor modules.

    Attributes
    ----------
    _model   : fitted sklearn estimator, or None if not yet loaded.
    _scaler  : fitted StandardScaler, or None if not yet loaded.
    _meta    : metadata dict loaded from anomaly_metadata.json.
    """

    def __init__(self) -> None:
        self._model:  Any | None = None
        self._scaler: Any | None = None
        self._meta:   dict[str, Any] = {}
        self._buffers: dict[int, deque] = {1: deque(maxlen=ROLLING_WINDOW), 2: deque(maxlen=ROLLING_WINDOW)}

    # ------------------------------------------------------------------
    # load()
    # ------------------------------------------------------------------
    def load(self) -> bool:
        """
        Load the model and metadata from disk.

        Returns
        -------
        True if successful, False if the model file does not exist.
        """
        model, scaler = load_model()
        if model is None:
            return False
        self._model  = model
        self._scaler = scaler
        self._meta   = load_metadata()
        log.info(
            "ModelManager: loaded %s (trained_at=%s, synthetic=%s)",
            self._meta.get("model_type", "unknown"),
            self._meta.get("trained_at", "?"),
            self._meta.get("synthetic", "?"),
        )
        return True

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------
    def predict(
        self,
        temp: float,
        hum: float,
        light: float,
        node_id: int,  # accepted for future per-node context; not used in scaling
    ) -> tuple[bool, float]:
        """
        Score a single sensor reading.

        Parameters
        ----------
        temp, hum, light : sensor values.
        node_id          : originating node (reserved for per-node context).

        Returns
        -------
        (anomalous: bool, score: float)
        """
        if self._model is None or self._scaler is None:
            if not self.load():
                log.warning(
                    "ModelManager.predict called before model is available; "
                    "returning (False, 0.0)."
                )
                return False, 0.0

        if node_id not in self._buffers:
            self._buffers[node_id] = deque(maxlen=ROLLING_WINDOW)
        self._buffers[node_id].append([float(temp), float(hum), float(light)])

        raw_window = np.array(list(self._buffers[node_id]), dtype=float)
        eng_window = engineer_features(raw_window, window=ROLLING_WINDOW)
        eng_latest = eng_window[-1:]  # Single row with accurate rolling features matching training
        eng_sc = self._scaler.transform(eng_latest)

        raw_pred  = self._model.predict(eng_sc)[0]
        anomalous = bool(raw_pred == -1)

        try:
            df    = float(self._model.decision_function(eng_sc)[0])
            score = float(np.clip(0.5 - df / 2.0, 0.0, 1.0))
        except AttributeError:
            score = 1.0 if anomalous else 0.0

        log.debug(
            "ModelManager.predict: node%d temp=%.1f hum=%.1f light=%.0f → "
            "anomalous=%s score=%.4f",
            node_id, temp, hum, light, anomalous, score,
        )
        return anomalous, round(score, 4)

    # ------------------------------------------------------------------
    # retrain()
    # ------------------------------------------------------------------
    def retrain(self, db_path: str) -> bool:
        """
        Load recent readings from an existing HEALNET SQLite database and
        retrain the model.

        The database schema is defined in storage.py.  We query *both* node
        tables using the Storage helper, then pass the results through the
        full training pipeline.

        Parameters
        ----------
        db_path : absolute or relative path to healnet.db.

        Returns
        -------
        True if retraining succeeded, False on error.
        """
        _sklearn_required()

        if not os.path.isfile(db_path):
            log.error("retrain: database not found at %s", db_path)
            return False

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT node_id, ts, temp, hum, light, anomaly
                  FROM readings
                 WHERE temp IS NOT NULL
                   AND hum  IS NOT NULL
                   AND light IS NOT NULL
                 ORDER BY ts ASC
                 LIMIT 50000
                """
            ).fetchall()
            conn.close()
        except Exception as exc:
            log.error("retrain: failed to read database: %s", exc)
            return False

        if len(rows) < 50:
            log.warning(
                "retrain: only %d usable rows in DB (need ≥50) — aborting.", len(rows)
            )
            return False

        X_raw = np.array([[r["temp"], r["hum"], r["light"]] for r in rows], dtype=float)
        y_raw = np.array([int(bool(r["anomaly"])) for r in rows], dtype=int)

        log.info(
            "retrain: %d samples from DB (%d anomalous)", len(X_raw), int(y_raw.sum())
        )

        try:
            result = run_training_pipeline(X_raw=X_raw, y_raw=y_raw, synthetic=False)
            self._model  = result["best_model"]
            self._scaler = result["scaler"]
            self._meta   = load_metadata()
            log.info(
                "retrain: complete. Best model: %s (F1=%.4f)",
                result["best_result"]["model_name"],
                result["best_result"]["f1"],
            )
            return True
        except Exception as exc:
            log.error("retrain: pipeline failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # summary()
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """
        Return a summary dict for the currently loaded model.

        Keys
        ----
        training_samples : int
        model_type       : str
        trained_at       : str (ISO-8601 UTC)
        synthetic        : bool
        evaluation       : dict (precision, recall, f1, accuracy, fpr, fnr)
        loaded           : bool
        """
        loaded = self._model is not None
        if not loaded:
            return {
                "loaded":          False,
                "model_type":      None,
                "trained_at":      None,
                "training_samples": 0,
                "synthetic":       None,
                "evaluation":      {},
            }

        meta = self._meta
        return {
            "loaded":           True,
            "model_type":       meta.get("model_type"),
            "trained_at":       meta.get("trained_at"),
            "training_samples": meta.get("training_samples", 0),
            "synthetic":        meta.get("synthetic"),
            "evaluation":       meta.get("evaluation", {}),
        }


# ============================================================================
#  SECTION 10 – __main__ entry point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="HEALNET ML Anomaly Detection Pipeline")
    parser.add_argument("--train", action="store_true", help="Train models on synthetic or database readings and persist the best")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate/display stored model metrics and evaluation report")
    parser.add_argument("--retrain", type=str, metavar="DB_PATH", help="Retrain using readings from specified SQLite database")
    parser.add_argument("--n-normal", type=int, default=1500, help="Number of normal synthetic samples (default: 1500)")
    parser.add_argument("--n-abnormal", type=int, default=150, help="Number of abnormal synthetic samples (default: 150)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if args.evaluate and not args.train and not args.retrain:
        log.info("=" * 60)
        log.info("HEALNET ml_pipeline  —  evaluation report")
        log.info("=" * 60)
        if not os.path.isfile(EVAL_REPORT_PATH) and not os.path.isfile(METADATA_PATH):
            log.warning("No evaluation report or metadata found at %s. Please train first with --train.", EVAL_REPORT_PATH)
            os._exit(1)

        if os.path.isfile(EVAL_REPORT_PATH):
            with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            print(f"\nEvaluation Report (generated: {rep.get('timestamp')})")
            print(f"Synthetic: {rep.get('synthetic')} | Train: {rep.get('training_samples')} | Test: {rep.get('test_samples')}")
            print_comparison_table(rep.get("models_evaluated", []))
        elif os.path.isfile(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            print(f"\nMetadata Model: {meta.get('model_type')}")
            print(f"Evaluation: {meta.get('evaluation')}")
        os._exit(0)

    if args.retrain:
        log.info("=" * 60)
        log.info("HEALNET ml_pipeline  —  retrain from database %s", args.retrain)
        log.info("=" * 60)
        mm = ModelManager()
        ok = mm.retrain(args.retrain)
        if ok:
            log.info("Retraining successful.")
        else:
            log.error("Retraining failed.")
            os._exit(1)
        os._exit(0)

    # Default action or --train: run training pipeline
    log.info("=" * 60)
    log.info("HEALNET ml_pipeline  —  full training run")
    log.info("=" * 60)

    t0 = time.perf_counter()

    pipeline_result = run_training_pipeline(
        n_normal=args.n_normal,
        n_abnormal=args.n_abnormal,
    )

    elapsed = time.perf_counter() - t0
    log.info("Pipeline completed in %.2f s", elapsed)

    # Smoke-test predict()
    log.info("Smoke-testing predict() ...")
    normal_sample   = {"temp": 24.0, "hum": 55.0, "light": 1800.0}
    abnormal_sample = {"temp": 55.0, "hum": 95.0, "light": 4095.0}

    a_normal, s_normal     = predict(normal_sample)
    a_abnormal, s_abnormal = predict(abnormal_sample)

    print(f"  Normal sample   -> anomalous={a_normal},   score={s_normal:.4f}")
    print(f"  Abnormal sample -> anomalous={a_abnormal}, score={s_abnormal:.4f}")

    # Smoke-test ModelManager
    log.info("Smoke-testing ModelManager ...")
    mm = ModelManager()
    if mm.load():
        print("\n  ModelManager.summary():")
        for k, v in mm.summary().items():
            print(f"    {k}: {v}")

        is_anom, score = mm.predict(24.0, 55.0, 1800.0, node_id=1)
        print(f"\n  ModelManager.predict(normal) -> anomalous={is_anom}, score={score:.4f}")

        is_anom2, score2 = mm.predict(55.0, 95.0, 4095.0, node_id=2)
        print(f"  ModelManager.predict(fault)  -> anomalous={is_anom2}, score={score2:.4f}")
    else:
        log.warning("ModelManager could not load model - skipping predict smoke-test.")

    log.info("Done.")

    os._exit(0)
