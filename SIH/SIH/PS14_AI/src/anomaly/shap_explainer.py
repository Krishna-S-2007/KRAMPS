"""
SHAP Explainability — PS14 AI Cyber Defence Platform.

Generates SHAP feature attribution explanations for the XGBoost detection model.

Design principles:
  - Feature names are derived from the trained model artifact, never hard-coded.
  - SHAP computation is optional and isolated; failures do not propagate to prediction.
  - Background sampling is configurable via the shap section in config.yaml.
  - Explanation generation is run after prediction; it does not affect scores.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for XGBoost detection model.

    Usage:
        explainer = SHAPExplainer(model=detector.model, top_k=3)
        explanations = explainer.explain(X)   # list of strings, one per row
    """

    def __init__(self, model, top_k: int = 3, background_samples: int = 100):
        """
        Args:
            model:              Trained XGBoost model object (xgb.XGBClassifier).
            top_k:              Number of top contributing features to include
                                in the explanation string per row.
            background_samples: Maximum background samples for SHAP explainer.
                                Reduces memory usage on large datasets.
        """
        self.model = model
        self.top_k = top_k
        self.background_samples = background_samples
        self._explainer = None

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def explain(self, X: pd.DataFrame) -> List[str]:
        """
        Compute SHAP values and return top-K feature explanations per row.

        If SHAP computation fails for any reason (e.g. model incompatibility,
        missing library version), returns a list of "N/A" strings. This ensures
        the main detection pipeline is never blocked by explanation failures.

        Args:
            X: Feature DataFrame. Must have named columns (same as training).

        Returns:
            List of explanation strings, one per row in X.
            Format: "feat1 (+0.123), feat2 (-0.056), feat3 (+0.034)"
        """
        if self.model is None:
            logger.warning("[SHAPExplainer] No model provided. Returning N/A.")
            return ["N/A"] * len(X)

        try:
            return self._compute_explanations(X)
        except Exception as exc:
            logger.warning(
                f"[SHAPExplainer] Explanation generation failed: {exc}. "
                f"Detection results are unaffected."
            )
            return ["N/A"] * len(X)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _compute_explanations(self, X: pd.DataFrame) -> List[str]:
        """Initialize explainer (if needed) and compute SHAP values."""
        try:
            import shap
        except ImportError:
            logger.warning(
                "[SHAPExplainer] 'shap' library not installed. "
                "Install with: pip install shap"
            )
            return ["N/A"] * len(X)

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)

        shap_values = self._explainer.shap_values(X)

        # For XGBoost binary classification, shap_values may be a list [neg, pos]
        # or a single array. We want the positive (attack) class values.
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]

        feature_names: List[str] = list(X.columns)
        return self._format_top_k(shap_values, feature_names)

    def _format_top_k(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> List[str]:
        """Format top-K features by absolute SHAP value for each row."""
        k = min(self.top_k, len(feature_names))
        results = []
        for row_shap in shap_values:
            top_indices = np.argsort(np.abs(row_shap))[::-1][:k]
            parts = [
                f"{feature_names[i]} ({row_shap[i]:+.3f})"
                for i in top_indices
            ]
            results.append(", ".join(parts))
        return results
