"""
XGBoost Supervised Detector — PS14 AI Cyber Defence Platform.

Design principles:
  - Hyperparameters are fully injected at construction time (from config.yaml).
  - SMOTE oversampling is applied ONLY to the training split (never the full dataset).
  - Feature ordering at inference is validated against the training feature list.
  - Model artifacts include metadata (feature names, params, training timestamp).
  - No dataset-specific assumptions, labels, or column names are embedded.
  - Prediction never retrains the model.
"""

import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class XGBoostDetector:
    """
    Supervised binary classifier for network attack detection using XGBoost.

    Outputs:
      - attack_probability: continuous float in [0.0, 1.0].
      - binary prediction: 0 (benign) or 1 (attack) based on configurable threshold.

    Models must be trained and saved before calling prediction methods.
    Prediction never triggers retraining.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        eval_metric: str = "logloss",
        use_smote: bool = True,
        train_test_split_ratio: float = 0.2,
    ):
        """
        Args:
            n_estimators:           Number of boosting rounds.
            max_depth:              Maximum tree depth.
            learning_rate:          Step size shrinkage.
            subsample:              Fraction of training rows per tree.
            colsample_bytree:       Fraction of features per tree.
            random_state:           Random seed for determinism.
            eval_metric:            XGBoost evaluation metric.
            use_smote:              Apply SMOTE on the training split only.
            train_test_split_ratio: Validation fraction. SMOTE never sees val data.
        """
        self.params: Dict = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "eval_metric": eval_metric,
        }
        self.use_smote = use_smote
        self.train_test_split_ratio = train_test_split_ratio
        self.model: Optional[xgb.XGBClassifier] = None
        self._feature_names: Optional[List[str]] = None
        self._is_trained: bool = False
        self._trained_at: Optional[str] = None

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Return the list of features the model was trained on."""
        if self._feature_names is not None:
            return self._feature_names
        if self.model is not None:
            if hasattr(self.model, "feature_names_in_"):
                return list(self.model.feature_names_in_)
            try:
                booster = self.model.get_booster()
                if booster and booster.feature_names:
                    return booster.feature_names
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Train the XGBoost detector.

        Pipeline:
          1. Train / validation split (val data is NEVER oversampled).
          2. SMOTE on training split only (if enabled and class imbalance exists).
          3. XGBoost training with early evaluation on validation set.

        Args:
            X: Feature DataFrame (numeric; encoded by FeatureEncoder).
            y: Binary integer label Series (0=benign, 1=attack).

        Returns:
            (X_val, y_val, val_probs): Validation set and predicted probabilities
            for use in downstream evaluation (not performed here).

        Raises:
            ValueError: If X or y are empty, or classes are missing.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Cannot train on empty dataset.")
        if len(X) != len(y):
            raise ValueError(
                f"Feature matrix rows ({len(X)}) do not match label count ({len(y)})."
            )

        self._feature_names = list(X.columns)

        # ── Step 1: Split before any oversampling ───────────────────────
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.train_test_split_ratio,
            random_state=self.params["random_state"],
            stratify=y if len(y.unique()) > 1 else None,
        )
        logger.info(
            f"[XGBoostDetector] Train: {len(X_train):,} | Val: {len(X_val):,}"
        )

        # ── Step 2: SMOTE on training split only ────────────────────────
        if self.use_smote and len(y_train.unique()) > 1:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.params["random_state"])
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(
                    f"[XGBoostDetector] SMOTE applied → {len(X_train):,} training samples."
                )
            except ImportError:
                logger.warning(
                    "[XGBoostDetector] imbalanced-learn not installed. "
                    "SMOTE skipped. Install with: pip install imbalanced-learn"
                )
        elif self.use_smote and len(y_train.unique()) == 1:
            logger.warning(
                "[XGBoostDetector] SMOTE requested but only one class in training split. "
                "Skipping SMOTE."
            )

        # ── Step 3: Train XGBoost ────────────────────────────────────────
        self._feature_names = list(X_train.columns)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._is_trained = True
        self._trained_at = datetime.now(timezone.utc).isoformat()

        # Update feature names reliably from the trained model if available
        if hasattr(self.model, "feature_names_in_"):
            self._feature_names = list(self.model.feature_names_in_)
        else:
            try:
                booster = self.model.get_booster()
                if booster and booster.feature_names:
                    self._feature_names = booster.feature_names
            except Exception:
                pass

        val_probs = self.model.predict_proba(X_val)[:, 1]
        attack_rate_train = float(y_train.mean())
        logger.info(
            f"[XGBoostDetector] Training complete. "
            f"Attack rate in training: {attack_rate_train:.2%}. "
            f"Val samples: {len(X_val):,}."
        )
        return X_val, y_val, val_probs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_attack_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return continuous attack probability for each row in X.

        Args:
            X: Feature DataFrame. Must match training feature schema.

        Returns:
            numpy array of float values in [0.0, 1.0].

        Raises:
            RuntimeError: If model has not been trained/loaded.
            ValueError:   If feature columns are incompatible.
        """
        self._assert_ready()
        X_aligned = self._align_inference_features(X)
        probs = self.model.predict_proba(X_aligned)
        return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary prediction (0=benign, 1=attack) using given threshold.

        Args:
            X:         Feature DataFrame.
            threshold: Decision boundary on attack_probability.

        Returns:
            numpy int array of 0s and 1s.
        """
        probs = self.predict_attack_probability(X)
        return (probs >= threshold).astype(int)

    def get_feature_importances(self) -> pd.Series:
        """
        Return XGBoost feature importance scores as a sorted Series.

        Raises:
            RuntimeError: If model is not trained.
        """
        self._assert_ready()
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model, feature list, params, and metadata to a pickle.

        Args:
            filepath: Destination path.
        """
        self._assert_ready()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifact = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
            "is_trained": self._is_trained,
            "trained_at": self._trained_at,
            "use_smote": self.use_smote,
        }
        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)
        logger.info(f"[XGBoostDetector] Model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model artifact.

        Args:
            filepath: Path to model pickle file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError:        If artifact is missing required keys.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"XGBoost model artifact not found: {filepath}\n"
                f"Run training first: python main.py --mode train --data <path>"
            )
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)

        required = {"model", "feature_names", "params", "is_trained"}
        missing = required - set(checkpoint.keys())
        if missing:
            raise ValueError(
                f"Model artifact at '{filepath}' is incomplete (missing: {missing}). "
                f"Retrain to regenerate."
            )

        self.model = checkpoint["model"]
        self._feature_names = checkpoint["feature_names"]
        self.params = checkpoint["params"]
        self._is_trained = checkpoint["is_trained"]
        self._trained_at = checkpoint.get("trained_at", "unknown")
        self.use_smote = checkpoint.get("use_smote", False)
        logger.info(
            f"[XGBoostDetector] Loaded model from: {filepath} "
            f"(trained: {self._trained_at}, features: {len(self._feature_names or [])})."
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if not self._is_trained or self.model is None:
            raise RuntimeError(
                "XGBoostDetector has not been trained or loaded. "
                "Call train() or load_model() first."
            )

    def _align_inference_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align inference columns to training feature order. Raises on critical mismatch."""
        feat_names = self.feature_names
        if feat_names is None:
            return X

        training_set = set(feat_names)
        inference_set = set(X.columns)

        extra = inference_set - training_set
        missing = training_set - inference_set

        if extra:
            logger.warning(
                f"[XGBoostDetector] {len(extra)} extra column(s) at inference dropped: {sorted(extra)}"
            )
            X = X.drop(columns=list(extra))

        if missing:
            raise ValueError(
                f"Inference feature schema mismatch. "
                f"Missing required training features: {sorted(missing)}"
            )

        return X[feat_names]
