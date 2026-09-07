r"""
PS14 End-to-End Offline Test
=============================
Tests the full pipeline WITHOUT Kafka or Docker:
  Simulator scenario engine  -->  TelemetryAdapter  -->  ML inference

Run from PS14-AI-Cyber-Defence root:

    python test_e2e.py --ml-root <path_to_PS14_AI> [--events 20] [--scenario ddos]

Example:
    python test_e2e.py --ml-root "c:\Users\Krishna\Desktop\SIH\SIH\SIH\PS14_AI"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os
import logging
import math
from datetime import datetime, timezone

# ── Make PS14-AI-Cyber-Defence importable ─────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

logging.basicConfig(
    level=logging.WARNING,           # suppress verbose ML logs during demo
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("PS14.E2E")

# ── Inline minimal TelemetryEvent (no Kafka / pydantic needed) ───────────────
class _Event:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, mode="json"):
        d = dict(self.__dict__)
        if isinstance(d.get("timestamp"), datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        return d


# ── Minimal asset / topology stubs (reuse the real ones if available) ─────────
def _build_assets():
    return [
        {"asset_id": "OPS-01",  "ip": "10.0.1.10", "type": "workstation"},
        {"asset_id": "OPS-02",  "ip": "10.0.1.11", "type": "workstation"},
        {"asset_id": "SRV-01",  "ip": "10.0.2.10", "type": "server"},
        {"asset_id": "SRV-02",  "ip": "10.0.2.11", "type": "server"},
        {"asset_id": "DNS-01",  "ip": "10.0.3.10", "type": "dns"},
        {"asset_id": "GW-01",   "ip": "10.0.0.1",  "type": "gateway"},
    ]


def _build_connections():
    return [
        {"source": "OPS-01", "destination": "SRV-01", "protocols": ["TCP", "UDP"], "ports": [443, 80, 22]},
        {"source": "OPS-02", "destination": "SRV-01", "protocols": ["TCP"],        "ports": [443, 22]},
        {"source": "OPS-01", "destination": "DNS-01", "protocols": ["UDP"],        "ports": [53]},
        {"source": "OPS-02", "destination": "DNS-01", "protocols": ["UDP"],        "ports": [53]},
        {"source": "SRV-01", "destination": "GW-01",  "protocols": ["TCP"],        "ports": [443, 80]},
        {"source": "SRV-02", "destination": "GW-01",  "protocols": ["TCP"],        "ports": [443]},
    ]


def generate_event(assets, connections, scenario: str) -> _Event:
    asset_map = {a["asset_id"]: a for a in assets}
    conn = random.choice(connections)
    src  = asset_map[conn["source"]]
    dst  = asset_map[conn["destination"]]
    proto = random.choice(conn["protocols"])
    dport = random.choice(conn["ports"])

    tcp_flags = dns_query = dns_entropy = http_method = http_status = None

    if proto == "TCP":
        tcp_flags = random.choice(["A", "PA", "FA"])
    if dport == 53:
        dns_query   = random.choice(["ops.internal", "command.internal"])
        dns_entropy = round(random.uniform(1.5, 3.0), 2)
    if dport in [80, 443]:
        http_method = random.choice(["GET", "POST"])
        http_status = random.choice([200, 200, 204])

    ev = _Event(
        timestamp       = datetime.now(timezone.utc),
        source_ip       = src["ip"],
        destination_ip  = dst["ip"],
        source_port     = random.randint(49152, 65535),
        destination_port= dport,
        protocol        = proto,
        duration        = 1.0,
        bytes_in        = 1000,
        bytes_out       = 1000,
        packets_in      = 10,
        packets_out     = 10,
        tcp_flags       = tcp_flags,
        dns_query       = dns_query,
        dns_entropy     = dns_entropy,
        http_method     = http_method,
        http_status     = http_status,
    )

    # ── Apply scenario profiles ───────────────────────────────────────────────
    if scenario == "ddos":
        ev.destination_ip = "10.0.2.10"
        ev.destination_port = 443
        ev.packets_out = random.randint(500, 2000)
        ev.packets_in  = random.randint(5, 20)
        ev.bytes_out   = ev.packets_out * 64
        ev.bytes_in    = ev.packets_in  * 1500
        ev.duration    = round(random.uniform(0.01, 0.1), 3)
        ev.tcp_flags   = "S"

    elif scenario == "brute_force":
        ev.destination_port = random.choice([22, 3389, 5900])
        ev.packets_out = random.randint(5, 20)
        ev.packets_in  = random.randint(1, 5)
        ev.bytes_out   = random.randint(200, 800)
        ev.bytes_in    = random.randint(100, 400)
        ev.duration    = round(random.uniform(0.1, 0.5), 3)
        ev.tcp_flags   = random.choice(["S", "FA", "RST"])

    elif scenario == "reconnaissance":
        ev.destination_port = random.choice([22, 80, 443, 8080, 3389, 445])
        ev.packets_out = random.randint(1, 3)
        ev.packets_in  = 0
        ev.bytes_out   = random.randint(40, 80)
        ev.bytes_in    = 0
        ev.duration    = round(random.uniform(0.001, 0.01), 4)
        ev.tcp_flags   = "S"

    elif scenario == "exfiltration":
        ev.destination_port = random.choice([443, 8443, 4444])
        ev.packets_out = random.randint(1000, 5000)
        ev.packets_in  = random.randint(5, 20)
        ev.bytes_out   = random.randint(500_000, 5_000_000)
        ev.bytes_in    = random.randint(1000, 5000)
        ev.duration    = round(random.uniform(10.0, 120.0), 2)
        ev.tcp_flags   = "PA"

    elif scenario == "dns_c2":
        ev.destination_ip   = "10.0.3.10"
        ev.destination_port = 53
        ev.protocol         = "UDP"
        ev.dns_query        = "c2." + "".join(random.choices("abcdefghij", k=12)) + ".evil.io"
        ev.dns_entropy      = round(random.uniform(3.5, 5.0), 2)
        ev.packets_out = random.randint(2, 10)
        ev.packets_in  = random.randint(1, 5)
        ev.bytes_out   = random.randint(80, 300)
        ev.bytes_in    = random.randint(60, 200)
        ev.duration    = round(random.uniform(0.5, 2.0), 3)

    elif scenario == "lateral_movement":
        ev.destination_port = random.choice([445, 135, 139, 3389])
        ev.packets_out = random.randint(20, 100)
        ev.packets_in  = random.randint(15, 80)
        ev.bytes_out   = random.randint(5000, 50000)
        ev.bytes_in    = random.randint(3000, 30000)
        ev.duration    = round(random.uniform(0.5, 5.0), 3)
        ev.tcp_flags   = random.choice(["PA", "A"])

    elif scenario == "communication_disruption":
        ev.destination_port = 443
        ev.packets_out = random.randint(1, 5)
        ev.packets_in  = 0
        ev.bytes_out   = random.randint(40, 200)
        ev.bytes_in    = 0
        ev.duration    = round(random.uniform(0.001, 0.05), 4)
        ev.tcp_flags   = "RST"

    else:  # normal
        ev.duration   = round(random.uniform(0.2, 5.0), 3)
        ev.packets_out = random.randint(3, 40)
        ev.packets_in  = random.randint(2, 30)
        ev.bytes_out   = random.randint(500, 50000)
        ev.bytes_in    = random.randint(300, 30000)

    return ev


def load_ml_pipeline(ml_root: str):
    ml_root = os.path.abspath(ml_root)
    if ml_root not in sys.path:
        sys.path.insert(0, ml_root)

    import yaml
    from src.features    import FeatureEncoder
    from src.detection   import XGBoostDetector
    from src.anomaly     import AnomalyDetector
    from src.fusion      import ScoreFusion

    config_path = os.path.join(ml_root, "config", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sup_dir = os.path.join(ml_root, config["paths"]["supervised_model_dir"])
    ano_dir = os.path.join(ml_root, config["paths"]["anomaly_model_dir"])

    encoder = FeatureEncoder.load(os.path.join(sup_dir, "preprocessor.pkl"))
    detector = XGBoostDetector()
    detector.load_model(os.path.join(sup_dir, "xgboost_model.pkl"))

    anomaly_encoder = FeatureEncoder.load(os.path.join(ano_dir, "anomaly_preprocessor.pkl"))
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_model(
        os.path.join(ano_dir, "anomaly_model.pkl"),
        os.path.join(ano_dir, "normalization.json"),
    )

    fusion_cfg = config.get("fusion", {})
    fusion = ScoreFusion(
        detection_weight=fusion_cfg.get("detection_weight", 0.6),
        anomaly_weight=fusion_cfg.get("anomaly_weight", 0.4),
    )
    threshold = config["model"].get("decision_threshold", 0.5)
    return detector, anomaly_detector, fusion, encoder, anomaly_encoder, threshold


def infer(event_dict, adapter, detector, anomaly_detector, fusion, encoder, anomaly_encoder):
    df, ctx = adapter.process_event(event_dict, stateful=True)

    X_sup = encoder.transform(df)
    X_sup = X_sup.drop(columns=[encoder.target_col], errors="ignore")
    attack_prob = float(detector.predict_attack_probability(X_sup)[0])

    X_ano = anomaly_encoder.transform(df)
    X_ano = X_ano.drop(columns=[anomaly_encoder.target_col], errors="ignore")
    anomaly_score = float(anomaly_detector.predict_anomaly_score(X_ano)[0])

    alert_score = float(fusion.fuse([attack_prob], [anomaly_score])[0])
    return attack_prob, anomaly_score, alert_score, ctx


SCENARIO_WEIGHTS = {
    "normal":                  75,
    "ddos":                     4,
    "brute_force":              5,
    "reconnaissance":           6,
    "exfiltration":             2,
    "dns_c2":                   3,
    "lateral_movement":         3,
    "communication_disruption": 2,
}

SCENARIO_COLORS = {
    "normal":                  "",
    "ddos":                    "\033[91m",   # red
    "brute_force":             "\033[91m",
    "reconnaissance":          "\033[93m",   # yellow
    "exfiltration":            "\033[95m",   # magenta
    "dns_c2":                  "\033[95m",
    "lateral_movement":        "\033[94m",   # blue
    "communication_disruption":"\033[93m",
}
RESET = "\033[0m"


def run(ml_root: str, n_events: int, forced_scenario=None):
    try:
        from ml_engine.streaming.telemetry_adapter import TelemetryAdapter
    except ImportError:
        from ml.telemetry_adapter import TelemetryAdapter

    assets      = _build_assets()
    connections = _build_connections()
    adapter     = TelemetryAdapter(window_seconds=30.0)

    print()
    print("=" * 95)
    print("  PS14 END-TO-END OFFLINE TEST (Stateful Behavioral Window)")
    print("  Simulator -> Stateful Window -> FeatureAggregator -> XGBoost + IsolationForest -> Alert")
    print("=" * 95)
    print(f"  ML root  : {ml_root}")
    print(f"  Events   : {n_events}")
    print(f"  Mode     : {'FORCED ' + forced_scenario.upper() if forced_scenario else 'RANDOM MIX'}")
    print()
    print("  Loading ML pipeline...", end="", flush=True)

    detector, anomaly_detector, fusion, encoder, anomaly_encoder, threshold = load_ml_pipeline(ml_root)

    print(" DONE")
    print()
    print(f"  {'SCENARIO':<24} {'WINDOW':<16} {'SRC IP':<14} {'DST IP':<14} {'PORT':<5} {'PROTO':<4} "
          f"{'ATTACK':>7} {'ANOMALY':>8} {'ALERT':>7}  STATUS")
    print("  " + "-" * 120)

    alert_count  = 0
    total        = 0
    scenario_stats = {s: {"count": 0, "alerts": 0} for s in SCENARIO_WEIGHTS}

    for _ in range(n_events):
        if forced_scenario:
            scenario = forced_scenario
        else:
            scenario = random.choices(
                list(SCENARIO_WEIGHTS.keys()),
                weights=list(SCENARIO_WEIGHTS.values()),
                k=1
            )[0]

        event = generate_event(assets, connections, scenario)
        event_dict = event.model_dump()

        attack_prob, anomaly_score, alert_score, ctx = infer(
            event_dict, adapter, detector, anomaly_detector, fusion, encoder, anomaly_encoder
        )

        is_alert = alert_score >= threshold
        total += 1
        scenario_stats[scenario]["count"] += 1
        if is_alert:
            alert_count += 1
            scenario_stats[scenario]["alerts"] += 1

        color  = SCENARIO_COLORS.get(scenario, "")
        status = f"{color}!! ALERT{RESET}" if is_alert else "   OK   "
        w_str = f"n={ctx.get('window_event_count', 1)} ({ctx.get('window_span_s', 0.0):.1f}s)"

        print(
            f"  {color}{scenario:<24}{RESET} "
            f"{w_str:<16} "
            f"{event_dict.get('source_ip','?'):<14} "
            f"{event_dict.get('destination_ip','?'):<14} "
            f"{event_dict.get('destination_port','?'):<5} "
            f"{event_dict.get('protocol','?'):<4} "
            f"{attack_prob:>7.3f} "
            f"{anomaly_score:>8.3f} "
            f"{alert_score:>7.3f}  "
            f"{status}"
        )

    print()
    print("=" * 80)
    print(f"  SUMMARY  |  Total events: {total}  |  Alerts fired: {alert_count}  "
          f"({100*alert_count/max(total,1):.1f}%)  |  Threshold: {threshold}")
    print()
    print(f"  {'SCENARIO':<30} {'EVENTS':>8} {'ALERTS':>8} {'ALERT%':>8}")
    print("  " + "-" * 58)
    for sc, st in scenario_stats.items():
        if st["count"] > 0:
            pct = 100 * st["alerts"] / st["count"]
            flag = " <-- ATTACK" if sc != "normal" and st["alerts"] > 0 else ""
            print(f"  {sc:<30} {st['count']:>8} {st['alerts']:>8} {pct:>7.1f}%{flag}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    DEFAULT_ML_PATHS = [
        os.path.abspath(os.path.join(THIS_DIR, "ml_engine")),
        os.path.abspath(os.path.join(THIS_DIR, "..", "SIH", "SIH", "PS14_AI")),
        os.path.abspath(os.path.join(THIS_DIR, "..", "PS14_AI")),
        os.path.abspath(os.path.join(THIS_DIR, "ml")),
    ]
    default_ml_root = next((p for p in DEFAULT_ML_PATHS if os.path.isdir(p) and os.path.exists(os.path.join(p, "models"))), DEFAULT_ML_PATHS[0])

    parser = argparse.ArgumentParser(description="PS14 End-to-End Offline Test")
    parser.add_argument(
        "--ml-root",
        default=default_ml_root,
        help=f'Path to ml_engine directory (default: {default_ml_root})',
    )
    parser.add_argument(
        "--events",
        type=int,
        default=30,
        help="Number of events to generate and infer. Default: 30",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_WEIGHTS.keys()),
        default=None,
        help="Force a single scenario (default: random weighted mix).",
    )
    args = parser.parse_args()
    run(ml_root=args.ml_root, n_events=args.events, forced_scenario=args.scenario)

