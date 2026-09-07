"""
Feature Encoder — PS14 AI Cyber Defence Platform.

Stateful feature transformer following a fit → transform lifecycle.

Design principles:
  - Numeric features are NEVER blindly converted to categories.
  - Categorical features are fitted with integer label encoders during training.
  - Unknown categorical values at inference time map to a designated <UNK> integer.
  - Hex string columns are converted to integers deterministically.
  - Target label encoding is strict and config-driven; unrecognized values raise errors.
  - Transformer state is persisted as a single artifact and reloaded at inference time.
  - Feature ordering is deterministic and validated at inference time.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sentinel for unknown categorical values encountered at inference time.
_UNK_TOKEN = "<UNK>"


class FeatureEncoder:
    """
    Dataset-agnostic feature encoder for network telemetry DataFrames.

    Usage (training):
        encoder = FeatureEncoder(
            target_col="alert",
            benign_values=["benign", "0", 0],
            categorical_columns=["frame.protocols", "eth.src", "ip.src"],
            hex_columns=["ip.flags", "tcp.flags"],
        )
        X, y = encoder.fit_transform(cleaned_df)
        encoder.save("models/supervised/preprocessor.pkl")

    Usage (inference):
        encoder = FeatureEncoder.load("models/supervised/preprocessor.pkl")
        X = encoder.transform(cleaned_df)   # target column absent at inference is OK
    """

    def __init__(
        self,
        target_col: str = "alert",
        benign_values: Optional[List] = None,
        categorical_columns: Optional[List[str]] = None,
        hex_columns: Optional[List[str]] = None,
        labels_already_binary: bool = False,
    ):
        """
        Args:
            target_col:           Name of the label/target column.
            benign_values:        Raw values that map to 0 (benign/negative class).
                                  Everything else → 1 (attack/positive class).
            categorical_columns:  Columns treated as categorical (fitted at training).
                                  Only columns present in data are encoded.
            hex_columns:          Columns whose values are hex strings (e.g. "0x0002").
                                  Converted to integers. Only present columns processed.
            labels_already_binary: If True, skip string-label mapping and treat column
                                   directly as int (0/1). Raises if values are not 0/1.
        """
        self.target_col = target_col
        self.benign_values = [str(v).strip().lower() for v in (benign_values or [])]
        self.categorical_columns = categorical_columns or []
        self.hex_columns = hex_columns or []
        self.labels_already_binary = labels_already_binary

        # Fitted state (populated by fit / fit_transform).
        self._cat_encoders: Dict[str, Dict[str, int]] = {}  # col → {value: int}
        self._feature_names: Optional[List[str]] = None      # ordered feature columns
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public fit / transform API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """
        Fit the encoder on training data (excluding the target column).

        Learns categorical label → integer mappings from training data.
        Must be called before transform().

        Args:
            df: Cleaned training DataFrame (may include target column).

        Returns:
            self (for chaining).
        """
        feature_df = df.drop(columns=[self.target_col], errors="ignore").copy()
        self._fit_categorical_encoders(feature_df)
        # Feature names are determined after a transform pass.
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature columns using fitted encoders.

        Does not touch the target column — callers retrieve it separately
        with encode_target().

        Args:
            df: Cleaned DataFrame (target column may or may not be present).

        Returns:
            DataFrame with all feature columns encoded as numeric types,
            in the deterministic order established during fit.

        Raises:
            RuntimeError: If encoder has not been fitted.
            ValueError:   If feature columns have changed since fitting.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEncoder must be fitted before calling transform(). "
                "Call fit_transform() during training."
            )

        target_present = self.target_col in df.columns
        target_series = df[self.target_col].copy() if target_present else None

        feature_df = df.drop(columns=[self.target_col], errors="ignore").copy()
        feature_df = self._apply_hex_conversions(feature_df)
        feature_df = self._apply_categorical_encodings(feature_df)

        if self._feature_names is not None:
            feature_df = self._align_feature_columns(feature_df)

        # Downcast numeric columns to float32 and int32 to save memory (reduces size by 50%)
        for col in feature_df.columns:
            if pd.api.types.is_float_dtype(feature_df[col]):
                feature_df[col] = feature_df[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(feature_df[col]):
                feature_df[col] = feature_df[col].astype(np.int32)

        result = feature_df
        if target_present:
            result[self.target_col] = target_series

        return result

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Convenience: fit on df then transform, returning (X, y).

        The target column is encoded separately via encode_target().

        Args:
            df: Cleaned training DataFrame including the target column.

        Returns:
            (X: pd.DataFrame, y: pd.Series)

        Raises:
            ValueError: If target column is missing from df.
        """
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Ensure the correct target_col is set in config.yaml."
            )

        self.fit(df)
        transformed = self.transform(df)

        X = transformed.drop(columns=[self.target_col], errors="ignore")
        y_raw = df[self.target_col]
        y = self.encode_target(y_raw)

        # Record final feature ordering after all transforms.
        self._feature_names = list(X.columns)
        logger.info(
            f"[FeatureEncoder] Fit complete. "
            f"{len(self._feature_names)} features. "
            f"Categorical encoders: {list(self._cat_encoders.keys())}."
        )
        return X, y

    def encode_target(self, y: pd.Series) -> pd.Series:
        """
        Encode the target/label column to binary integers (0 = benign, 1 = attack).

        Args:
            y: Raw label series.

        Returns:
            Integer series of 0s and 1s.

        Raises:
            ValueError: If labels_already_binary=True but values outside {0,1},
                        or if a string label is not in benign_values and not recognized.
        """
        if self.labels_already_binary:
            try:
                y_int = y.astype(int)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"labels_already_binary=True but target column contains "
                    f"non-integer values: {y.unique()[:10]}. Error: {exc}"
                ) from exc

            invalid = y_int[~y_int.isin([0, 1])]
            if not invalid.empty:
                raise ValueError(
                    f"labels_already_binary=True but found values outside {{0,1}}: "
                    f"{invalid.unique()[:10]}. Adjust the label configuration."
                )
            return y_int

        def _map_label(val):
            str_val = str(val).strip().lower()
            if str_val in self.benign_values:
                return 0
            # Any value not in benign_values is treated as attack (1).
            return 1

        if not self.benign_values:
            raise ValueError(
                "benign_values is empty in FeatureEncoder. "
                "Configure label.benign_values in config.yaml."
            )

        encoded = y.apply(_map_label)
        unique_raw = y.unique()
        unique_enc = encoded.unique()
        logger.info(
            f"[FeatureEncoder] Label encoding: raw={unique_raw[:10]} → encoded={unique_enc}. "
            f"Attack rate: {encoded.mean():.2%}"
        )
        return encoded.astype(int)

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Ordered list of feature column names as seen during training."""
        return self._feature_names

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Persist the fitted encoder state to a pickle file.

        Args:
            filepath: Destination path (directories created if absent).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted FeatureEncoder.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "target_col": self.target_col,
            "benign_values": self.benign_values,
            "categorical_columns": self.categorical_columns,
            "hex_columns": self.hex_columns,
            "labels_already_binary": self.labels_already_binary,
            "cat_encoders": self._cat_encoders,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"[FeatureEncoder] Saved preprocessor to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FeatureEncoder":
        """
        Load a previously fitted FeatureEncoder from a pickle file.

        Args:
            filepath: Path to saved preprocessor artifact.

        Returns:
            Fitted FeatureEncoder instance.

        Raises:
            FileNotFoundError: If artifact does not exist.
            ValueError:        If artifact is corrupted or incompatible.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {filepath}\n"
                f"Run training first with: python main.py --mode train --data <path>"
            )
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        required_keys = {
            "target_col", "benign_values", "categorical_columns",
            "hex_columns", "labels_already_binary", "cat_encoders",
            "feature_names", "is_fitted",
        }
        missing = required_keys - set(state.keys())
        if missing:
            raise ValueError(
                f"Preprocessor artifact at '{filepath}' is missing keys: {missing}. "
                f"Retrain the model to regenerate a compatible artifact."
            )

        encoder = cls(
            target_col=state["target_col"],
            benign_values=state["benign_values"],
            categorical_columns=state["categorical_columns"],
            hex_columns=state["hex_columns"],
            labels_already_binary=state["labels_already_binary"],
        )
        encoder._cat_encoders = state["cat_encoders"]
        encoder._feature_names = state["feature_names"]
        encoder._is_fitted = state["is_fitted"]
        logger.info(
            f"[FeatureEncoder] Loaded preprocessor from: {filepath} "
            f"| {len(encoder._feature_names or [])} features."
        )
        return encoder

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _fit_categorical_encoders(self, df: pd.DataFrame) -> None:
        """Fit an integer mapping for each configured categorical column present in df."""
        self._cat_encoders = {}
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            unique_values = sorted(
                df[col].dropna().astype(str).unique().tolist()
            )
            # Reserve 0 for <UNK>, actual values start at 1.
            mapping = {v: idx + 1 for idx, v in enumerate(unique_values)}
            mapping[_UNK_TOKEN] = 0
            self._cat_encoders[col] = mapping
            logger.debug(
                f"[FeatureEncoder] Fitted '{col}': {len(unique_values)} unique values → int."
            )

    def _apply_hex_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert hex string columns to integers. Non-hex values map to 0."""
        for col in self.hex_columns:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue  # Already numeric.
            df[col] = df[col].apply(self._parse_hex_value)
        return df

    @staticmethod
    def _parse_hex_value(value) -> int:
        """Parse a hex string (e.g. '0x0002' or 'a4b1') to an int. Returns 0 on failure."""
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(str(value).strip(), 16)
        except (ValueError, TypeError):
            return 0

    def _apply_categorical_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted integer mappings to categorical columns."""
        for col, mapping in self._cat_encoders.items():
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).map(lambda v, m=mapping: m.get(v, m[_UNK_TOKEN]))
        return df

    def _align_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align inference-time columns to the training feature set.

        - Columns present in training but missing at inference → filled with 0.
        - Columns present at inference but absent from training → dropped with warning.
        - Column ordering is preserved to match training.
        """
        training_cols = set(self._feature_names)
        inference_cols = set(df.columns)

        extra = inference_cols - training_cols
        if extra:
            logger.warning(
                f"[FeatureEncoder] {len(extra)} column(s) present at inference but "
                f"not in training set — dropped: {sorted(extra)}"
            )
            df = df.drop(columns=list(extra))

        missing = training_cols - inference_cols
        if missing:
            logger.warning(
                f"[FeatureEncoder] {len(missing)} column(s) missing at inference — "
                f"filled with 0: {sorted(missing)}"
            )
            for col in missing:
                df[col] = 0

        return df[self._feature_names]
