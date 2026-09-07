"""
Telemetry Cleaning — PS14 AI Cyber Defence Platform.

This module cleans raw telemetry DataFrames before feature encoding.

Design principles:
  - All column names are derived from configuration, never hard-coded here.
  - Missing configured columns are silently skipped (not hard errors).
  - Numeric, categorical, and timestamp columns are distinguished explicitly.
  - No dataset-specific protocol maps, IP ranges, or field names are embedded.
  - Missing value handling is configurable.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TelemetryCleaner:
    """
    Dataset-agnostic cleaner for raw network telemetry DataFrames.

    Actions:
      1. Drop configured columns (silently skips absent ones).
      2. Parse and decompose a timestamp column into numeric parts.
      3. Drop rows with excessive missing values.
      4. Impute remaining numeric NaN values (median/mean/zero).

    No field-specific assumptions (protocol maps, IP ranges, MAC tables)
    live in this class. Feature-level encoding is handled by FeatureEncoder.
    """

    def __init__(
        self,
        drop_columns: Optional[List[str]] = None,
        timestamp_col: Optional[str] = None,
        max_nan_fraction: float = 0.5,
        numeric_impute_strategy: str = "median",
        handle_infinity: bool = True,
    ):
        """
        Args:
            drop_columns:            Columns to remove (skipped if absent).
            timestamp_col:           Column name containing timestamps.
                                     Set to None/empty if not present.
            max_nan_fraction:        Rows where more than this fraction of
                                     columns are NaN are dropped.
            numeric_impute_strategy: How to fill remaining numeric NaNs.
                                     Options: "median", "mean", "zero".
            handle_infinity:         If True, replace inf/-inf with NaN before imputation.
        """
        self.drop_columns = drop_columns or []
        self.timestamp_col = timestamp_col or ""
        self.max_nan_fraction = max_nan_fraction
        self.numeric_impute_strategy = numeric_impute_strategy
        self.handle_infinity = handle_infinity

        if numeric_impute_strategy not in ("median", "mean", "zero"):
            raise ValueError(
                f"numeric_impute_strategy must be 'median', 'mean', or 'zero'; "
                f"got '{numeric_impute_strategy}'."
            )

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the full cleaning pipeline on a raw telemetry DataFrame.

        Args:
            df: Raw telemetry DataFrame.

        Returns:
            Cleaned DataFrame with numeric/temporal features ready for encoding.

        Raises:
            ValueError: If the result is empty after cleaning.
        """
        if df is None or len(df) == 0:
            raise ValueError("Cannot clean an empty DataFrame.")

        df = df.copy()
        
        # Replace inf and -inf with NaN if configured
        if self.handle_infinity:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
        original_rows = len(df)

        # 1. Drop configured columns (silently ignore absent ones).
        df = self._drop_configured_columns(df)

        # 2. Parse timestamp column into numeric temporal features.
        df = self._decompose_timestamp(df)

        # 3. Drop rows exceeding the maximum NaN fraction.
        df = self._drop_excessive_nan_rows(df)

        # 4. Impute remaining numeric NaN values.
        df = self._impute_numeric_nans(df)

        df.reset_index(drop=True, inplace=True)
        dropped = original_rows - len(df)
        logger.info(
            f"[TelemetryCleaner] {original_rows:,} rows → {len(df):,} rows "
            f"({dropped:,} dropped). Columns: {len(df.columns)}."
        )

        if len(df) == 0:
            raise ValueError(
                "All rows were removed during cleaning. "
                "Check drop_columns and max_nan_fraction configuration."
            )

        return df

    # ------------------------------------------------------------------
    # Private Steps
    # ------------------------------------------------------------------

    def _drop_configured_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.drop_columns if c in df.columns]
        absent  = [c for c in self.drop_columns if c not in df.columns]
        if absent:
            logger.debug(f"[TelemetryCleaner] Configured drop columns absent: {absent}")
        if present:
            df = df.drop(columns=present)
            logger.debug(f"[TelemetryCleaner] Dropped {len(present)} columns: {present}")
        return df

    def _decompose_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the configured timestamp column into 6 integer sub-features:
        year, month, day, hour, minute, second.

        Rows with completely unparseable timestamps are filled with 0.
        The original timestamp column is then dropped.
        """
        col = self.timestamp_col
        if not col or col not in df.columns:
            if col:
                logger.debug(f"[TelemetryCleaner] Timestamp column '{col}' not found — skipped.")
            return df

        ts = pd.to_datetime(df[col], errors="coerce", utc=False)
        df["frame.year"]   = ts.dt.year.fillna(0).astype(int)
        df["frame.month"]  = ts.dt.month.fillna(0).astype(int)
        df["frame.day"]    = ts.dt.day.fillna(0).astype(int)
        df["frame.hour"]   = ts.dt.hour.fillna(0).astype(int)
        df["frame.minute"] = ts.dt.minute.fillna(0).astype(int)
        df["frame.second"] = ts.dt.second.fillna(0).astype(int)
        df = df.drop(columns=[col])
        logger.debug(f"[TelemetryCleaner] Parsed timestamp column '{col}' into 6 temporal features.")
        return df

    def _drop_excessive_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where the fraction of NaN values exceeds max_nan_fraction."""
        nan_fractions = df.isnull().mean(axis=1)
        mask = nan_fractions <= self.max_nan_fraction
        removed = (~mask).sum()
        if removed:
            logger.debug(
                f"[TelemetryCleaner] Dropped {removed:,} rows with >{self.max_nan_fraction*100:.0f}% NaN."
            )
        return df[mask]

    def _impute_numeric_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute remaining NaN values in numeric columns using the configured strategy."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df

        if self.numeric_impute_strategy == "median":
            fill_vals = df[numeric_cols].median()
        elif self.numeric_impute_strategy == "mean":
            fill_vals = df[numeric_cols].mean()
        else:  # "zero"
            fill_vals = pd.Series(0, index=numeric_cols)

        df[numeric_cols] = df[numeric_cols].fillna(fill_vals)
        return df
