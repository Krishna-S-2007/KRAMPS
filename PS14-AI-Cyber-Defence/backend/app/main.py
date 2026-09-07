import os
import sys
import json
import asyncio
import random
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Ensure repo root and ML root are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Auto-locate PS14_AI
DEFAULT_ML_PATHS = [
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "SIH", "SIH", "PS14_AI")),
    os.path.abspath(os.path.join(PROJECT_ROOT, "..", "PS14_AI")),
    os.path.abspath(os.path.join(PROJECT_ROOT, "ml")),
]
ML_ROOT = next((p for p in DEFAULT_ML_PATHS if os.path.isdir(p) and os.path.exists(os.path.join(p, "models"))), DEFAULT_ML_PATHS[0])
if ML_ROOT not in sys.path:
    sys.path.insert(0, ML_ROOT)

from backend.app.telemetry.schemas import TelemetryEvent
from ml.telemetry_adapter import TelemetryAdapter
from simulator.network.loader import load_assets, load_topology
from simulator.scenarios.normal import apply_normal_profile
from simulator.scenarios.ddos import apply_ddos_profile, TARGET_ASSET as DDOS_TARGET
from simulator.scenarios.reconnaissance import apply_recon_profile
from simulator.scenarios.brute_force import apply_brute_force_profile, SOURCE_ASSET as BRUTE_SOURCE, TARGET_ASSET as BRUTE_TARGET
from simulator.scenarios.lateral_movement import apply_lateral_movement_profile, SOURCE_ASSET as LATERAL_SOURCE, LATERAL_TARGETS
from simulator.scenarios.exfiltration import apply_exfiltration_profile, SOURCE_ASSET as EXFIL_SOURCE
from simulator.scenarios.dns_c2 import apply_dns_c2_profile, SOURCE_ASSET as DNS_C2_SOURCE, DNS_ASSET
from simulator.scenarios.communication_disruption import apply_communication_disruption_profile, TARGET_ASSET as COMM_TARGET

logger = logging.getLogger("PS14.Server")

# =============================================================================
# Global Pipeline State & Loader
# =============================================================================

class PipelineEngine:
    def __init__(self, ml_root: str):
        self.ml_root = ml_root
        self.adapter = TelemetryAdapter()
        self.detector = None
        self.anomaly_detector = None
        self.fusion = None
        self.encoder = None
        self.anomaly_encoder = None
        self.threshold = 0.5
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        try:
            import yaml
            from src.features import FeatureEncoder
            from src.detection import XGBoostDetector
            from src.anomaly import AnomalyDetector
            from src.fusion import ScoreFusion

            config_path = os.path.join(self.ml_root, "config", "config.yaml")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"Config not found at {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            sup_dir = os.path.join(self.ml_root, config["paths"]["supervised_model_dir"])
            ano_dir = os.path.join(self.ml_root, config["paths"]["anomaly_model_dir"])

            self.encoder = FeatureEncoder.load(os.path.join(sup_dir, "preprocessor.pkl"))
            self.detector = XGBoostDetector()
            self.detector.load_model(os.path.join(sup_dir, "xgboost_model.pkl"))

            self.anomaly_encoder = FeatureEncoder.load(os.path.join(ano_dir, "anomaly_preprocessor.pkl"))
            self.anomaly_detector = AnomalyDetector()
            self.anomaly_detector.load_model(
                os.path.join(ano_dir, "anomaly_model.pkl"),
                os.path.join(ano_dir, "normalization.json")
            )

            fusion_cfg = config.get("fusion", {})
            self.fusion = ScoreFusion(
                detection_weight=fusion_cfg.get("detection_weight", 0.6),
                anomaly_weight=fusion_cfg.get("anomaly_weight", 0.4),
            )
            self.threshold = float(config.get("model", {}).get("decision_threshold", 0.5))
            self.is_loaded = True
            logger.info(f"[PipelineEngine] ML models successfully loaded from {self.ml_root}")
        except Exception as e:
            logger.error(f"[PipelineEngine] Failed to load ML models: {e}")
            self.is_loaded = False

    def predict_event(self, event_dict: Dict[str, Any], stateful: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.is_loaded:
            raise RuntimeError("ML Pipeline models are not loaded.")

        # 1. Normalization & Stateful Flow/Window Aggregation
        df, ctx = self.adapter.process_event(event_dict, stateful=stateful)

        # 2. Supervised XGBoost Inference
        X_sup = self.encoder.transform(df)
        X_sup = X_sup.drop(columns=[self.encoder.target_col], errors="ignore")
        attack_prob = float(self.detector.predict_attack_probability(X_sup)[0])

        # 3. Unsupervised Isolation Forest Anomaly Inference
        X_ano = self.anomaly_encoder.transform(df)
        X_ano = X_ano.drop(columns=[self.anomaly_encoder.target_col], errors="ignore")
        anomaly_score = float(self.anomaly_detector.predict_anomaly_score(X_ano)[0])

        # 4. Score Fusion
        alert_score = float(self.fusion.fuse([attack_prob], [anomaly_score])[0])

        predictions = {
            "attack_probability": round(attack_prob, 4),
            "anomaly_score": round(anomaly_score, 4),
            "alert_score": round(alert_score, 4),
            "is_alert": bool(alert_score >= self.threshold),
            "threshold": self.threshold
        }
        return predictions, ctx

    def reset(self):
        self.adapter.reset()

engine = PipelineEngine(ML_ROOT)

# =============================================================================
# In-Memory Event Storage & Stream Manager
# =============================================================================

class SimulationManager:
    def __init__(self):
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        self.forced_scenario: Optional[str] = None
        self.interval: float = 0.5
        self.history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.subscribers: List[asyncio.Queue] = []
        self.stats = {
            "total_events": 0,
            "total_alerts": 0,
            "scenarios": {}
        }
        self.assets = load_assets()
        self.topology = load_topology()
        self.asset_map = {a["asset_id"]: a for a in self.assets}

    def generate_base_event(self) -> TelemetryEvent:
        conn = random.choice(self.topology["connections"])
        source = self.asset_map[conn["source"]]
        dest = self.asset_map[conn["destination"]]
        proto = random.choice(conn["protocols"])
        dport = random.choice(conn["ports"])

        tcp_flags = dns_query = dns_entropy = http_method = http_status = None
        if proto == "TCP":
            tcp_flags = random.choice(["A", "PA", "FA"])
        if dport == 53:
            dns_query = random.choice(["ops.internal", "command.internal", "files.internal"])
            dns_entropy = round(random.uniform(1.5, 3.0), 2)
        if dport in [80, 443]:
            http_method = random.choice(["GET", "POST"])
            http_status = random.choice([200, 200, 200, 204])

        return TelemetryEvent(
            timestamp=datetime.now(timezone.utc),
            source_ip=source["ip"],
            destination_ip=dest["ip"],
            source_port=random.randint(49152, 65535),
            destination_port=dport,
            protocol=proto,
            duration=1.0,
            bytes_in=1000,
            bytes_out=1000,
            packets_in=10,
            packets_out=10,
            tcp_flags=tcp_flags,
            dns_query=dns_query,
            dns_entropy=dns_entropy,
            http_method=http_method,
            http_status=http_status,
        )

    def apply_scenario(self, event: TelemetryEvent, scenario: str) -> TelemetryEvent:
        if scenario == "normal":
            return apply_normal_profile(event)
        elif scenario == "ddos":
            return apply_ddos_profile(event, self.asset_map[DDOS_TARGET]["ip"])
        elif scenario == "reconnaissance":
            return apply_recon_profile(event, self.asset_map["OPS-01"]["ip"])
        elif scenario == "brute_force":
            return apply_brute_force_profile(event, self.asset_map[BRUTE_SOURCE], self.asset_map[BRUTE_TARGET])
        elif scenario == "lateral_movement":
            return apply_lateral_movement_profile(event, self.asset_map[LATERAL_SOURCE], self.asset_map[random.choice(LATERAL_TARGETS)])
        elif scenario == "exfiltration":
            return apply_exfiltration_profile(event, self.asset_map[EXFIL_SOURCE])
        elif scenario == "dns_c2":
            return apply_dns_c2_profile(event, self.asset_map[DNS_C2_SOURCE], self.asset_map[DNS_ASSET])
        elif scenario == "communication_disruption":
            return apply_communication_disruption_profile(event, self.asset_map[COMM_TARGET])
        return event

    def _worker(self):
        scenario_weights = {
            "normal": 75, "reconnaissance": 6, "brute_force": 5,
            "ddos": 4, "lateral_movement": 3, "dns_c2": 3,
            "exfiltration": 2, "communication_disruption": 2
        }
        scenarios = list(scenario_weights.keys())
        weights = list(scenario_weights.values())

        while self.is_running:
            try:
                scenario = self.forced_scenario or random.choices(scenarios, weights=weights, k=1)[0]
                event = self.generate_base_event()
                event = self.apply_scenario(event, scenario)
                event_dict = event.model_dump(mode="json")

                # ML Inference with stateful window
                if engine.is_loaded:
                    preds, ctx = engine.predict_event(event_dict, stateful=True)
                else:
                    preds = {"attack_probability": 0.0, "anomaly_score": 0.0, "alert_score": 0.0, "is_alert": False}
                    ctx = {"window_event_count": 1, "window_span_s": 0.0}

                combined = {
                    "id": len(self.history) + 1,
                    "scenario": scenario,
                    "event": event_dict,
                    "window_context": ctx,
                    "predictions": preds,
                    "timestamp": event_dict["timestamp"]
                }

                # Update stats
                self.stats["total_events"] += 1
                if preds.get("is_alert"):
                    self.stats["total_alerts"] += 1
                self.stats["scenarios"][scenario] = self.stats["scenarios"].get(scenario, 0) + 1

                # Update history buffer
                self.history.append(combined)
                if len(self.history) > self.max_history:
                    self.history.pop(0)

                # Observability print to terminal
                is_alert = preds.get("is_alert", False)
                status_str = "!! ALERT" if is_alert else "   OK   "
                src_ip = event_dict.get("source_ip", "?")
                dst_ip = event_dict.get("destination_ip", "?")
                dport = event_dict.get("destination_port", "?")
                proto = event_dict.get("protocol", "?")
                att = preds.get("attack_probability", 0.0)
                ano = preds.get("anomaly_score", 0.0)
                alt = preds.get("alert_score", 0.0)
                w_events = ctx.get("window_event_count", 1)
                w_span = ctx.get("window_span_s", 0.0)

                print(
                    f"[{scenario.upper():<24}] [WIN:{w_events:>2} events, {w_span:>4.1f}s] "
                    f"{src_ip:>15} -> {dst_ip:<15}:{dport:<5} {proto:<4} | "
                    f"att={att:.3f} ano={ano:.3f} alt={alt:.3f} | {status_str}",
                    flush=True
                )

                # Broadcast to SSE subscribers
                dead_subs = []
                for q in list(self.subscribers):
                    try:
                        q.put_nowait(combined)
                    except Exception:
                        dead_subs.append(q)
                for q in dead_subs:
                    if q in self.subscribers:
                        self.subscribers.remove(q)

                import time
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Simulation loop error: {e}")
                import time
                time.sleep(1.0)

    def start(self, scenario: Optional[str] = None, interval: float = 0.5):
        if not self.is_running:
            self.is_running = True
            self.forced_scenario = scenario
            self.interval = interval
            import threading
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            logger.info(f"Simulation started. Mode: {scenario or 'RANDOM'}, Interval: {interval}s")

    def stop(self):
        if self.is_running:
            self.is_running = False
            logger.info("Simulation stopped.")

    def reset(self):
        self.history.clear()
        self.stats = {"total_events": 0, "total_alerts": 0, "scenarios": {}}
        engine.reset()
        logger.info("Simulation manager and behavioral window reset.")

sim_manager = SimulationManager()

# =============================================================================
# FastAPI App Lifecycle & Routes
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PS14 AI Cyber Defence Server starting up...")
    sim_manager.start(scenario=None, interval=0.5)
    yield
    sim_manager.stop()
    logger.info("PS14 Server shutdown complete.")

app = FastAPI(
    title="PS14 - AI Cyber Defence Unified Server",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "system": "PS14 AI Cyber Defence Platform",
        "version": "2.0.0",
        "ml_models_loaded": engine.is_loaded,
        "simulation_running": sim_manager.is_running,
        "endpoints": [
            "/health",
            "/api/simulator/start",
            "/api/simulator/stop",
            "/api/simulator/reset",
            "/api/simulator/status",
            "/api/telemetry/recent",
            "/api/alerts",
            "/api/stream",
            "/api/predict"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ml_root": ML_ROOT,
        "ml_loaded": engine.is_loaded,
        "simulation_running": sim_manager.is_running,
        "total_events_processed": sim_manager.stats["total_events"]
    }

@app.post("/api/simulator/start")
def start_simulation(scenario: Optional[str] = None, interval: float = Query(0.5, ge=0.01, le=10.0)):
    sim_manager.start(scenario=scenario, interval=interval)
    return {
        "status": "started",
        "scenario": scenario or "RANDOM_MIX",
        "interval_seconds": interval
    }

@app.post("/api/simulator/stop")
def stop_simulation():
    sim_manager.stop()
    return {"status": "stopped"}

@app.post("/api/simulator/reset")
def reset_simulation():
    sim_manager.reset()
    return {"status": "reset", "message": "Flow state and simulation history cleared."}

@app.get("/api/simulator/status")
def simulator_status():
    return {
        "is_running": sim_manager.is_running,
        "forced_scenario": sim_manager.forced_scenario,
        "interval": sim_manager.interval,
        "stats": sim_manager.stats,
        "buffered_events": len(sim_manager.history)
    }

@app.get("/api/telemetry/recent")
def get_recent_telemetry(limit: int = Query(50, ge=1, le=500)):
    return list(reversed(sim_manager.history[-limit:]))

@app.get("/api/alerts")
def get_alerts(limit: int = Query(50, ge=1, le=500)):
    alerts = [item for item in sim_manager.history if item["predictions"].get("is_alert")]
    return list(reversed(alerts[-limit:]))

@app.post("/api/predict")
def predict_single_event(event: TelemetryEvent, stateful: bool = Query(True, description="Use stateful window tracking")):
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="ML pipeline models are not loaded")
    event_dict = event.model_dump(mode="json")
    preds, ctx = engine.predict_event(event_dict, stateful=stateful)
    return {
        "event": event_dict,
        "window_context": ctx,
        "predictions": preds
    }

@app.get("/api/stream")
async def sse_stream():
    queue = asyncio.Queue()
    sim_manager.subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in sim_manager.subscribers:
                sim_manager.subscribers.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")