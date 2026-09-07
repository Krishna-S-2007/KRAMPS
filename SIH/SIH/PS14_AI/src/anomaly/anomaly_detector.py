"""
Anomaly Detection — PS14 AI Cyber Defence Platform.

Isolation Forest-based unsupervised anomaly detection.

Design principles:
  - Trained ONLY on benign/normal baseline data when labels are available
    and `train_on_benign_only=True`. This is the recommended approach because
    Isolation Forest learns the boundary of normal behavior, not a mixture.
  - Normalization parameters (score_min, score_max) are computed from the
    TRAINING decision function scores and persisted with the model artifact.
  - At inference, the exact saved normalization parameters are reused.
    The inference batch is NEVER used to recompute normalization.
  - Prediction never retrains the model. Calling predict_anomaly_score()
    on an unfitted model raises RuntimeError.
  - No dataset-specific field assumptions or hard-coded thresholds.
"""

import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Isolation Forest anomaly detector with persistent score normalization.

    Anomaly score output is always in [0.0, 1.0] where:
      1.0 = highly anomalous (abnormal behavior)
      0.0 = highly normal

    Normalization uses the training set decision_function range:
        normalized = 1.0 - (raw_score - score_min) / (score_max - score_min + ε)

    This normalization is fitted once during training and reused deterministically
    at inference via saved parameters.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        max_samples: str = "auto",
        train_on_benign_only: bool = True,
    ):
        """
        Args:
            contamination:       Fraction of outliers in the training data.
            n_estimators:        Number of Isolation Forest trees.
            random_state:        Random seed for determinism.
            max_samples:         Samples per tree. 'auto' = min(256, n_samples).
            train_on_benign_only: If True, fit only on benign rows (label == 0).
                                  Recommended when labels are available at training time.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = max_samples
        self.train_on_benign_only = train_on_benign_only

        self._model: Optional[IsolationForest] = None
        self._score_min: Optional[float] = None  # Saved from training set.
        self._score_max: Optional[float] = None  # Saved from training set.
        self._is_fitted: bool = False
        self._trained_at: Optional[str] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AnomalyDetector":
        """
        Fit the Isolation Forest on training data.

        If train_on_benign_only=True and y is provided, the model is trained
        only on rows where y == 0 (benign). This is the recommended mode because
        Isolation Forest defines the normal boundary, and mixing attack samples
        into the training set degrades this boundary.

        If train_on_benign_only=False, the full X is used (unsupervised mode).
        If y is None, the full X is always used regardless of train_on_benign_only.

        Args:
            X: Feature DataFrame (same encoding as XGBoost input).
            y: Optional label Series (0=benign, 1=attack).
               Required when train_on_benign_only=True for correct baseline selection.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If X is empty, or if train_on_benign_only=True but
                        no benign samples are present.
        """
        if len(X) == 0:
            raise ValueError("Cannot fit AnomalyDetector on an empty dataset.")

        if self.train_on_benign_only and y is not None:
            benign_mask = y == 0
            X_fit = X[benign_mask]
            n_benign = benign_mask.sum()
            n_total = len(X)
            if n_benign == 0:
                raise ValueError(
                    "train_on_benign_only=True but no benign samples (label==0) "
                    "found in training data. Ensure labels are correctly encoded."
                )
            logger.info(
                f"[AnomalyDetector] Fitting on {n_benign:,} benign samples "
                f"(out of {n_total:,} total, {n_benign/n_total:.1%} benign)."
            )
        else:
            X_fit = X
            if self.train_on_benign_only and y is None:
                logger.warning(
                    "[AnomalyDetector] train_on_benign_only=True but y was not provided. "
                    "Fitting on all training samples (unsupervised mode)."
                )
            else:
                logger.info(
                    f"[AnomalyDetector] Fitting on all {len(X_fit):,} training samples "
                    f"(unsupervised mode)."
                )

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_samples=self.max_samples,
        )
        self._model.fit(X_fit)

        # ── Compute and persist normalization parameters from training data ──
        raw_scores = self._model.decision_function(X_fit)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        self._is_fitted = True
        self._trained_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"[AnomalyDetector] Fit complete. "
            f"Score normalization: min={self._score_min:.4f}, max={self._score_max:.4f}. "
            f"Trained at: {self._trained_at}."
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized anomaly score for each row in X.

        Uses the TRAINING normalization parameters (score_min, score_max)
        persisted during fit(). Does NOT recompute normalization from X.

        Args:
            X: Feature DataFrame with same column schema as training.

        Returns:
            numpy array of float in [0.0, 1.0]:
              1.0 = highly anomalous
              0.0 = highly normal

        Raises:
            RuntimeError: If model has not been fitted or loaded.
        """
        self._assert_ready()

        raw_scores = self._model.decision_function(X)
        eps = 1e-8
        score_range = self._score_max - self._score_min + eps
        normalized = 1.0 - (raw_scores - self._score_min) / score_range
        return np.clip(normalized, 0.0, 1.0)

    def predict_binary(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return binary predictions: -1 = anomaly, 1 = normal (native IsolationForest output).

        Returns:
            numpy array of -1 or 1.
        """
        self._assert_ready()
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: str, norm_filepath: str) -> None:
        """
        Save the Isolation Forest model as a pickle and the normalization parameters
        as a JSON file.

        Args:
            filepath:      Path for model pickle file.
            norm_filepath: Path for normalization parameters JSON file.

        Raises:
            RuntimeError: If model is not fitted.
        """
        self._assert_ready()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        os.makedirs(os.path.dirname(norm_filepath), exist_ok=True)
        
        # Save model pickle
        with open(filepath, "wb") as f:
            pickle.dump(self._model, f)
            
        # Save normalization parameters
        import json
        norm_data = {
            "score_min": self._score_min,
            "score_max": self._score_max
        }
        with open(norm_filepath, "w") as f:
            json.dump(norm_data, f, indent=2)
            
        logger.info(
            f"[AnomalyDetector] Saved model pickle to: {filepath} "
            f"and normalization params to: {norm_filepath}"
        )

    def load_model(self, filepath: str, norm_filepath: str) -> None:
        """
        Load Isolation Forest model from pickle and normalization parameters from JSON.

        Args:
            filepath:      Path to model pickle file.
            norm_filepath: Path to normalization JSON file.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Anomaly model artifact not found: {filepath}\n"
                f"Run training first: python main.py --mode train --data <path>"
            )
        if not os.path.isfile(norm_filepath):
            raise FileNotFoundError(
                f"Normalization JSON not found: {norm_filepath}\n"
                f"Run training first: python main.py --mode train --data <path>"
            )

        with open(filepath, "rb") as f:
            self._model = pickle.load(f)

        import json
        with open(norm_filepath, "r") as f:
            norm_data = json.load(f)

        self._score_min = norm_data["score_min"]
        self._score_max = norm_data["score_max"]
        self._is_fitted = True
        logger.info(
            f"[AnomalyDetector] Loaded model from: {filepath} and "
            f"normalization params from: {norm_filepath} (score range: [{self._score_min:.4f}, {self._score_max:.4f}])."
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "AnomalyDetector has not been fitted or loaded. "
                "Call fit() during training or load_model() before prediction."
            )
        if self._score_min is None or self._score_max is None:
            raise RuntimeError(
                "Anomaly normalization parameters are missing. "
                "Retrain to regenerate the artifact."
            )
