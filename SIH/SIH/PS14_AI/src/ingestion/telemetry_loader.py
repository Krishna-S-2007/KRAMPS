"""
Telemetry Ingestion Module — PS14 AI Cyber Defence Platform.

Provides a clean, format-agnostic boundary for loading raw telemetry data.
The downstream preprocessing and ML pipeline should never depend on how
telemetry was loaded, only on the resulting DataFrame schema.

Supported formats: CSV, JSON, JSONL, Parquet, Pandas DataFrame.
"""

import os
import json
import logging
import pandas as pd
from typing import Optional, Union

logger = logging.getLogger(__name__)


class TelemetryLoader:
    """
    Dataset-agnostic telemetry loader.

    Loads raw telemetry from CSV, JSON, JSONL, Parquet, or a
    Pandas DataFrame. Validates that the resulting DataFrame is
    non-empty before returning it to the pipeline.

    The telemetry schema is not assumed here. The downstream
    preprocessor and feature encoder handle schema-specific logic
    driven by config.yaml.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".parquet": "parquet",
    }

    def __init__(self, csv_sep: str = ",", encoding: str = "utf-8"):
        """
        Args:
            csv_sep:   CSV column separator. Defaults to comma.
            encoding:  File encoding. Defaults to utf-8.
        """
        self.csv_sep = csv_sep
        self.encoding = encoding

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def load(
        self,
        source: Union[str, pd.DataFrame],
        fmt: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load telemetry from a file path or an existing DataFrame.

        Args:
            source: Absolute or relative path to telemetry file,
                    or a pre-loaded pandas DataFrame.
            fmt:    Format override. If None, inferred from extension.
                    Accepted: "csv", "json", "jsonl", "parquet".

        Returns:
            pd.DataFrame containing raw telemetry rows.

        Raises:
            TypeError:     If source is neither a path nor a DataFrame.
            FileNotFoundError: If file path does not exist.
            ValueError:    If format is unsupported or DataFrame is empty.
            RuntimeError:  If file parsing fails.
        """
        if isinstance(source, pd.DataFrame):
            return self._validate(source, label="DataFrame input")

        if not isinstance(source, (str, os.PathLike)):
            raise TypeError(
                f"Expected file path (str/PathLike) or pd.DataFrame, "
                f"got {type(source).__name__}."
            )

        path = os.path.abspath(str(source))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Telemetry file not found: {path}\n"
                f"Please supply the path via --data <path> or config."
            )

        resolved_fmt = fmt or self._infer_format(path)
        logger.info(f"[TelemetryLoader] Loading {resolved_fmt.upper()} from: {path}")

        try:
            if resolved_fmt == "csv":
                df = pd.read_csv(path, sep=self.csv_sep, encoding=self.encoding, low_memory=False)
            elif resolved_fmt == "json":
                df = pd.read_json(path, encoding=self.encoding)
            elif resolved_fmt == "jsonl":
                df = pd.read_json(path, lines=True, encoding=self.encoding)
            elif resolved_fmt == "parquet":
                df = pd.read_parquet(path)
            else:
                raise ValueError(
                    f"Unsupported telemetry format '{resolved_fmt}'. "
                    f"Supported: {list(self.SUPPORTED_EXTENSIONS.values())}"
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Failed to parse telemetry file '{path}': {exc}"
            ) from exc

        return self._validate(df, label=path)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _infer_format(self, path: str) -> str:
        _, ext = os.path.splitext(path.lower())
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Cannot infer telemetry format from extension '{ext}'.\n"
                f"Supported extensions: {list(self.SUPPORTED_EXTENSIONS.keys())}\n"
                f"Pass fmt='csv' (or 'json'/'parquet') explicitly if needed."
            )
        return self.SUPPORTED_EXTENSIONS[ext]

    @staticmethod
    def _validate(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError(
                f"Telemetry source '{label}' produced an empty DataFrame. "
                f"Ensure the file contains valid rows."
            )
        logger.info(
            f"[TelemetryLoader] Loaded {len(df):,} rows × {len(df.columns)} columns."
        )
        return df
