"""
Score Fusion — PS14 AI Cyber Defence Platform.

Combines XGBoost attack_probability and Isolation Forest anomaly_score
into a single fused alert_score using configurable weighted averaging.

Design principles:
  - Weights are injected from config.yaml. No hard-coded values.
  - Validates that weights sum to 1.0 (raises ValueError otherwise).
  - Fusion is stateless and model-agnostic: it only combines numeric scores.
  - Output is clipped to [0.0, 1.0].
  - Signature detection is intentionally NOT part of this fusion layer.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ScoreFusion:
    """
    Weighted linear fusion of detection and anomaly scores.

    Config-driven weights (from config.yaml fusion section):
        detection_weight + anomaly_weight must equal 1.0.

    Output:
        alert_score = (detection_weight × attack_probability)
                    + (anomaly_weight   × anomaly_score)

    Both inputs must be numpy arrays (or scalars) in [0.0, 1.0].
    """

    def __init__(self, detection_weight: float = 0.6, anomaly_weight: float = 0.4):
        """
        Args:
            detection_weight: Weight for the supervised XGBoost attack_probability.
            anomaly_weight:   Weight for the unsupervised anomaly_score.

        Raises:
            ValueError: If weights do not sum to 1.0 (within 1e-6 tolerance).
        """
        total = detection_weight + anomaly_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"fusion.detection_weight ({detection_weight}) + "
                f"fusion.anomaly_weight ({anomaly_weight}) = {total:.6f}, "
                f"but must sum to exactly 1.0. "
                f"Update config.yaml fusion section."
            )
        self.detection_weight = detection_weight
        self.anomaly_weight = anomaly_weight
        logger.info(
            f"[ScoreFusion] Initialized: "
            f"detection={detection_weight:.2f}, anomaly={anomaly_weight:.2f}."
        )

    def fuse(
        self,
        attack_probability: np.ndarray,
        anomaly_score: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weighted fusion of attack_probability and anomaly_score.

        Args:
            attack_probability: XGBoost output — array in [0.0, 1.0].
            anomaly_score:      Isolation Forest output — array in [0.0, 1.0].

        Returns:
            alert_score: numpy array in [0.0, 1.0].
        """
        ap = np.asarray(attack_probability, dtype=float)
        as_ = np.asarray(anomaly_score, dtype=float)

        if ap.shape != as_.shape:
            raise ValueError(
                f"attack_probability shape {ap.shape} != "
                f"anomaly_score shape {as_.shape}. "
                f"Both arrays must have the same length."
            )

        alert_score = (self.detection_weight * ap) + (self.anomaly_weight * as_)
        return np.clip(alert_score, 0.0, 1.0)
