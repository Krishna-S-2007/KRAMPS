"""
PS14 Live ML Bridge
===================
Consumes raw telemetry events from the Kafka topic ps14.raw.simulation,
normalises them through the stateful TelemetryAdapter into CIC-IDS2017 feature rows,
and feeds them directly into the trained PS14_AI ML pipeline for
real-time inference.

Run from the PS14-AI-Cyber-Defence repo root:

    python -m ml.live_ml_bridge --ml-root <path_to_PS14_AI>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from ml.telemetry_adapter import TelemetryAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PS14.Bridge")


def _load_ml_pipeline(ml_root: str):
    """
    Dynamically load the PS14_AI pipeline components from ml_root.
    Returns (detector, anomaly_detector, fusion, encoder, anomaly_encoder, threshold).
    """
    ml_root = os.path.abspath(ml_root)
    if ml_root not in sys.path:
        sys.path.insert(0, ml_root)

    import yaml
    from src.features import FeatureEncoder
    from src.detection import XGBoostDetector
    from src.anomaly import AnomalyDetector
    from src.fusion import ScoreFusion

    config_path = os.path.join(ml_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sup_dir = os.path.join(ml_root, config["paths"]["supervised_model_dir"])
    ano_dir = os.path.join(ml_root, config["paths"]["anomaly_model_dir"])

    paths = {
        "xgboost_model":          os.path.join(sup_dir, "xgboost_model.pkl"),
        "preprocessor":           os.path.join(sup_dir, "preprocessor.pkl"),
        "anomaly_model":          os.path.join(ano_dir,  "anomaly_model.pkl"),
        "anomaly_preprocessor":   os.path.join(ano_dir,  "anomaly_preprocessor.pkl"),
        "anomaly_normalization":  os.path.join(ano_dir,  "normalization.json"),
    }

    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing ML artifact '{name}': {path}\n"
                f"Run: python main.py --mode train --data <csv>  (inside {ml_root})"
            )

    encoder = FeatureEncoder.load(paths["preprocessor"])
    detector = XGBoostDetector()
    detector.load_model(paths["xgboost_model"])

    anomaly_encoder = FeatureEncoder.load(paths["anomaly_preprocessor"])
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_model(paths["anomaly_model"], paths["anomaly_normalization"])

    fusion_cfg = config.get("fusion", {})
    fusion = ScoreFusion(
        detection_weight=fusion_cfg.get("detection_weight", 0.6),
        anomaly_weight=fusion_cfg.get("anomaly_weight", 0.4),
    )
    threshold = float(config.get("model", {}).get("decision_threshold", 0.5))

    logger.info("[Bridge] ML pipeline loaded from: %s", ml_root)
    return detector, anomaly_detector, fusion, encoder, anomaly_encoder, threshold


def _infer(
    event_dict: Dict[str, Any],
    adapter: TelemetryAdapter,
    detector: Any,
    anomaly_detector: Any,
    fusion: Any,
    encoder: Any,
    anomaly_encoder: Any,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """Run one event through the stateful window and PS14_AI predict pipeline."""
    # 1. State/Window Aggregation & CIC Feature Mapping
    df, ctx = adapter.process_event(event_dict, stateful=True)

    # 2. Supervised XGBoost Inference
    X_sup = encoder.transform(df)
    X_sup = X_sup.drop(columns=[encoder.target_col], errors="ignore")
    attack_prob = float(detector.predict_attack_probability(X_sup)[0])

    # 3. Anomaly Isolation Forest Inference
    X_ano = anomaly_encoder.transform(df)
    X_ano = X_ano.drop(columns=[anomaly_encoder.target_col], errors="ignore")
    anomaly_score = float(anomaly_detector.predict_anomaly_score(X_ano)[0])

    # 4. Score Fusion
    alert_score = float(fusion.fuse([attack_prob], [anomaly_score])[0])

    return attack_prob, anomaly_score, alert_score, ctx


def run_live_bridge(ml_root: str, kafka_servers: str = "localhost:9092", window_seconds: float = 30.0):
    """Main loop: consume Kafka -> stateful window adapt -> infer -> print alerts."""
    from kafka import KafkaConsumer

    adapter = TelemetryAdapter(window_seconds=window_seconds)
    detector, anomaly_detector, fusion, encoder, anomaly_encoder, threshold = _load_ml_pipeline(ml_root)

    consumer = KafkaConsumer(
        "ps14.raw.simulation",
        bootstrap_servers=kafka_servers,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="ps14-ml-bridge",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    print()
    print("=" * 80)
    print("  PS14 LIVE ML BRIDGE — Stateful Window Telemetry -> Normalise -> Infer")
    print(f"  Kafka           : {kafka_servers}")
    print(f"  Window (sec)    : {window_seconds}")
    print(f"  Alert threshold : {threshold}")
    print("  Press CTRL+C to stop.")
    print("=" * 80)
    print()

    for message in consumer:
        event = message.value

        try:
            attack_prob, anomaly_score, alert_score, ctx = _infer(
                event, adapter, detector, anomaly_detector, fusion, encoder, anomaly_encoder
            )
        except Exception as exc:
            logger.warning("[Bridge] Inference error: %s", exc)
            continue

        is_alert = alert_score >= threshold
        prefix = "!! ALERT" if is_alert else "   OK   "

        src = ctx.get("source_ip", "?")
        dst = ctx.get("destination_ip", "?")
        port = ctx.get("destination_port", "?")
        proto = ctx.get("protocol", "?")
        w_events = ctx.get("window_event_count", 1)
        w_span = ctx.get("window_span_s", 0.0)

        print(
            f"[{prefix}] "
            f"[WINDOW: events={w_events:<3} span={w_span:<5.2f}s] "
            f"{src:>15} -> {dst:<15}:{port:<5} {proto:<4} | "
            f"att={attack_prob:.3f} ano={anomaly_score:.3f} alt={alert_score:.3f}",
            flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PS14 Live ML Bridge",
        description="Connects the PS14 simulator to the PS14_AI ML pipeline in real time with stateful windowing.",
    )
    parser.add_argument(
        "--ml-root",
        required=True,
        help="Absolute path to the PS14_AI project directory (where main.py lives).",
    )
    parser.add_argument(
        "--kafka",
        default="localhost:9092",
        help="Kafka bootstrap servers. Default: localhost:9092",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=30.0,
        help="Rolling window duration in seconds (default: 30.0)",
    )
    args = parser.parse_args()
    run_live_bridge(ml_root=args.ml_root, kafka_servers=args.kafka, window_seconds=args.window)
