"""
Evaluation Package — PS14 AI Cyber Defence Platform.

NOTE: Evaluation is intended for use AFTER training on real telemetry data.
Do NOT interpret evaluation results as production model quality until
the system has been trained and evaluated on the actual operational dataset.
"""
from .metrics import evaluate_detection_performance

__all__ = ["evaluate_detection_performance"]
