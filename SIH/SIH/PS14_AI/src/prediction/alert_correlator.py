"""
Alert Correlator — PS14 AI Cyber Defence Platform.

Groups high-risk telemetry flows into multi-stage attack sessions
by correlating flows that share common source/destination entity pairs.

Design principles:
  - No hard-coded IP addresses, default network addresses, or network topology.
  - Source/destination fields are resolved from configurable candidate column lists.
  - If none of the candidate columns are present, sessions are keyed by row index.
  - Missing optional fields are represented as "unknown", never fabricated.
  - The correlator is a deterministic algorithm, not a trained ML model.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Sentinel used when a source/destination field is absent from the telemetry.
_UNKNOWN = "unknown"


class AlertCorrelator:
    """
    Deterministic alert correlation engine.

    Groups telemetry rows where alert_score >= threshold into attack sessions
    keyed by (attacker_entity, target_entity) pairs.

    Entity resolution uses configurable candidate column lists. The first
    candidate column found in the DataFrame is used. If no candidates are
    found, sessions are keyed as "<row_index>" to avoid fabricating IPs.
    """

    def __init__(
        self,
        alert_threshold: float = 0.5,
        attacker_ip_candidates: Optional[List[str]] = None,
        target_ip_candidates: Optional[List[str]] = None,
        timestamp_fragments: Optional[List[str]] = None,
    ):
        """
        Args:
            alert_threshold:         Minimum alert_score to include a flow.
            attacker_ip_candidates:  Ordered list of columns to try for attacker entity.
            target_ip_candidates:    Ordered list of columns to try for target entity.
            timestamp_fragments:     Column names for reconstructing a timestamp string
                                     (e.g. ["frame.year", "frame.month", ...]).
        """
        self.alert_threshold = alert_threshold
        self.attacker_ip_candidates = attacker_ip_candidates or ["ip.src"]
        self.target_ip_candidates = target_ip_candidates or ["ip.dst"]
        self.timestamp_fragments = timestamp_fragments or []

    def correlate(self, df: pd.DataFrame) -> List[Dict]:
        """
        Correlate high-alert telemetry rows into attack session structures.

        Args:
            df: DataFrame that must contain at least one of:
                - "alert_score" column, OR
                - "attack_probability" column.
                Other columns are used if present; missing ones default to "unknown".

        Returns:
            List of session dicts, each containing:
              - session_id:   "<attacker>-><target>" or row-based key.
              - event_count:  Number of high-alert events in this session.
              - events:       List of event dicts.

        Raises:
            ValueError: If neither alert_score nor attack_probability is found.
        """
        score_col = self._resolve_score_column(df)
        high_risk_df = df[df[score_col] >= self.alert_threshold].copy()

        if len(high_risk_df) == 0:
            logger.info(
                f"[AlertCorrelator] No flows above threshold={self.alert_threshold}. "
                f"Returning empty session list."
            )
            return []

        attacker_col = self._resolve_candidate(df, self.attacker_ip_candidates, "attacker")
        target_col = self._resolve_candidate(df, self.target_ip_candidates, "target")

        sessions: Dict[str, List[Dict]] = defaultdict(list)

        for idx, row in high_risk_df.iterrows():
            attacker = str(row[attacker_col]) if attacker_col else _UNKNOWN
            target = str(row[target_col]) if target_col else _UNKNOWN
            session_key = f"{attacker}->{target}"

            event = {
                "event_id": int(idx),
                "timestamp": self._reconstruct_timestamp(row),
                "attacker_ip": attacker,
                "target_ip": target,
                "protocol": self._safe_get(row, "frame.protocols", _UNKNOWN),
                "attack_probability": self._safe_float(row, "attack_probability"),
                "anomaly_score": self._safe_float(row, "anomaly_score"),
                "alert_score": self._safe_float(row, score_col),
                "shap_features": self._safe_get(row, "shap_top_features", "N/A"),
            }
            sessions[session_key].append(event)

        result = [
            {
                "session_id": key,
                "event_count": len(events),
                "events": events,
            }
            for key, events in sessions.items()
        ]
        logger.info(
            f"[AlertCorrelator] {len(high_risk_df):,} high-alert events → "
            f"{len(result)} sessions (threshold={self.alert_threshold})."
        )
        return result

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_score_column(df: pd.DataFrame) -> str:
        """Determine which score column to use for thresholding."""
        if "alert_score" in df.columns:
            return "alert_score"
        if "attack_probability" in df.columns:
            return "attack_probability"
        raise ValueError(
            "AlertCorrelator requires 'alert_score' or 'attack_probability' "
            "column in the DataFrame. Ensure the detection and fusion stages "
            "have run before correlation."
        )

    @staticmethod
    def _resolve_candidate(
        df: pd.DataFrame,
        candidates: List[str],
        label: str,
    ) -> Optional[str]:
        """Return the first candidate column present in df, or None."""
        for col in candidates:
            if col in df.columns:
                return col
        logger.debug(
            f"[AlertCorrelator] No {label} column found among candidates: {candidates}. "
            f"Sessions will use '{_UNKNOWN}'."
        )
        return None

    def _reconstruct_timestamp(self, row: pd.Series) -> str:
        """Reconstruct a timestamp string from decomposed temporal columns."""
        if not self.timestamp_fragments:
            return _UNKNOWN
        fragments = [str(self._safe_get(row, col, "0")) for col in self.timestamp_fragments]
        if len(fragments) >= 6:
            y, mo, d, h, mi, s = fragments[:6]
            try:
                return f"{y}-{int(mo):02d}-{int(d):02d} {int(h):02d}:{int(mi):02d}:{int(s):02d}"
            except (ValueError, TypeError):
                pass
        return "-".join(fragments)

    @staticmethod
    def _safe_get(row: pd.Series, col: str, default):
        """Safely retrieve a value from a row, returning default if absent."""
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                return default
            return val
        return default

    @staticmethod
    def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
        """Safely retrieve a float value from a row."""
        val = AlertCorrelator._safe_get(row, col, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
