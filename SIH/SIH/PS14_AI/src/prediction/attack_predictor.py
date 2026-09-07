"""
Attack Progression & Heuristic Risk Predictor — PS14 AI Cyber Defence Platform.

Produces per-session risk scores, MITRE ATT&CK tactic progression estimates,
time-to-next-phase estimates, and automated intervention recommendations
based on correlated alert session data.

IMPORTANT: This module implements HEURISTIC analysis, not a trained ML model.
The formulas are deterministic functions of observable session features.
All tuneable parameters are loaded from config.yaml (progression section).

A proper learned progression model will be developed once real training
telemetry is available. This heuristic layer serves as an interim
structural placeholder with clearly isolated logic.

Design principles:
  - No hard-coded IP addresses, targets, or network topology.
  - All configurable thresholds and weights read from the config dict.
  - Risk score is a bounded [0.0, 1.0] weighted combination of
    session-level alert statistics.
  - MITRE progression advances by observed event count.
  - Intervention text uses values from the session data, not invented strings.
  - Missing fields are handled explicitly — no silent defaults that fabricate data.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Defaults for all heuristic parameters. These are overridden by config.
_DEFAULT_CONFIG = {
    "risk_weight_avg_alert": 0.4,
    "risk_weight_max_alert": 0.4,
    "risk_weight_event_count": 0.2,
    "event_count_saturation": 10,
    "eta_base_minutes": 45,
    "eta_decay_per_event": 3.5,
    "eta_risk_decay": 20,
    "eta_minimum_minutes": 5,
    "risk_tier_critical": 0.8,
    "risk_tier_high": 0.5,
    "risk_tier_medium": 0.3,
    "events_per_stage": 2,
    "mitre_tactics": [
        "Reconnaissance",
        "Resource Development",
        "Initial Access",
        "Execution",
        "Persistence",
        "Privilege Escalation",
        "Defense Evasion",
        "Credential Access",
        "Discovery",
        "Lateral Movement",
        "Collection",
        "Command and Control",
        "Exfiltration",
        "Impact",
    ],
}


class AttackPredictor:
    """
    Heuristic attack progression and risk estimation engine.

    Consumes correlated session dictionaries produced by AlertCorrelator
    and outputs per-session risk estimates and progression state.

    NOTE: This is a deterministic heuristic, not a trained model.
    Results should be treated as indicative estimates until a proper
    learned model is trained on real telemetry data.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Dict matching the `progression` section of config.yaml.
                    Defaults are used for any missing key.
        """
        cfg = _DEFAULT_CONFIG.copy()
        if config:
            cfg.update(config)

        self.risk_weight_avg = float(cfg["risk_weight_avg_alert"])
        self.risk_weight_max = float(cfg["risk_weight_max_alert"])
        self.risk_weight_event = float(cfg["risk_weight_event_count"])
        self.event_count_saturation = int(cfg["event_count_saturation"])
        self.eta_base = float(cfg["eta_base_minutes"])
        self.eta_decay_per_event = float(cfg["eta_decay_per_event"])
        self.eta_risk_decay = float(cfg["eta_risk_decay"])
        self.eta_min = float(cfg["eta_minimum_minutes"])
        self.tier_critical = float(cfg["risk_tier_critical"])
        self.tier_high = float(cfg["risk_tier_high"])
        self.tier_medium = float(cfg["risk_tier_medium"])
        self.events_per_stage = max(1, int(cfg["events_per_stage"]))
        self.mitre_tactics: List[str] = list(cfg["mitre_tactics"])

        if not self.mitre_tactics:
            raise ValueError(
                "progression.mitre_tactics must be a non-empty list in config.yaml."
            )

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def predict_session_progression(self, session: Dict) -> Dict:
        """
        Compute heuristic risk and progression estimate for a single session.

        Args:
            session: Session dict from AlertCorrelator.correlate().
                     Expected keys: "session_id", "events" (list of event dicts).

        Returns:
            Dict containing:
              - session_id
              - event_count
              - risk         (float [0.0, 1.0])
              - current_tactic
              - next_tactic
              - target       (from session events, or "unknown")
              - ETA          (string, estimated minutes to next phase)
              - recommended_intervention
        """
        session_id = session.get("session_id", "unknown")
        events: List[Dict] = session.get("events", [])
        n_events = len(events)

        if n_events == 0:
            return {
                "session_id": session_id,
                "event_count": 0,
                "risk": 0.0,
                "current_tactic": self.mitre_tactics[0],
                "next_tactic": self.mitre_tactics[1] if len(self.mitre_tactics) > 1 else "N/A",
                "target": "unknown",
                "ETA": f"{int(self.eta_base)} mins",
                "recommended_intervention": "Monitor: no alert events in this session.",
            }

        # ── Risk Score (heuristic) ──────────────────────────────────────
        alert_scores = [float(e.get("alert_score", 0.0)) for e in events]
        avg_alert = float(np.mean(alert_scores))
        max_alert = float(np.max(alert_scores))
        event_saturation = min(n_events / self.event_count_saturation, 1.0)

        risk = min(1.0, (
            self.risk_weight_avg   * avg_alert
            + self.risk_weight_max   * max_alert
            + self.risk_weight_event * event_saturation
        ))

        # ── MITRE ATT&CK Tactic Progression ────────────────────────────
        n_tactics = len(self.mitre_tactics)
        # Advance one stage per `events_per_stage` events, capped at second-to-last.
        stage_idx = min(n_events // self.events_per_stage, n_tactics - 2)
        stage_idx = max(0, stage_idx)
        current_tactic = self.mitre_tactics[stage_idx]
        next_tactic = self.mitre_tactics[stage_idx + 1]

        # ── ETA to Next Phase ───────────────────────────────────────────
        eta_minutes = max(
            self.eta_min,
            self.eta_base
            - (n_events * self.eta_decay_per_event)
            - (risk * self.eta_risk_decay)
        )

        # ── Target Entity (from session data, not fabricated) ───────────
        target = str(events[0].get("target_ip", "unknown"))

        # ── Intervention Recommendation ─────────────────────────────────
        attacker = str(events[0].get("attacker_ip", "unknown"))
        intervention = self._generate_intervention(risk, target, attacker)

        return {
            "session_id": session_id,
            "event_count": n_events,
            "risk": round(float(risk), 4),
            "current_tactic": current_tactic,
            "next_tactic": next_tactic,
            "target": target,
            "ETA": f"{int(eta_minutes)} mins",
            "recommended_intervention": intervention,
        }

    def predict_all_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """
        Run progression estimation for all correlated sessions.

        Args:
            sessions: List of session dicts from AlertCorrelator.

        Returns:
            List of progression result dicts.
        """
        results = []
        for session in sessions:
            result = self.predict_session_progression(session)
            results.append(result)
        logger.info(
            f"[AttackPredictor] Processed {len(sessions)} sessions."
        )
        return results

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _generate_intervention(self, risk: float, target: str, attacker: str) -> str:
        """
        Generate a risk-tiered intervention recommendation.

        Uses session-derived target/attacker values. Never fabricates addresses.
        """
        if risk >= self.tier_critical:
            return (
                f"CRITICAL: Isolate host '{target}' and block source '{attacker}' immediately. "
                f"Escalate to incident response."
            )
        elif risk >= self.tier_high:
            return (
                f"HIGH: Apply active firewall rule for source '{attacker}' targeting '{target}'. "
                f"Reset credentials and audit access logs."
            )
        elif risk >= self.tier_medium:
            return (
                f"MEDIUM: Enable deep packet inspection on '{target}'. "
                f"Monitor '{attacker}' closely."
            )
        else:
            return (
                f"LOW: Log and monitor traffic from '{attacker}'. "
                f"No immediate action required."
            )
