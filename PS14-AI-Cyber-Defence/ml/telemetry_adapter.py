"""
PS14 Telemetry Adapter & Pipeline Bridge
========================================
Main entry point for converting raw telemetry JSON events into ML-compatible
78-feature vectors with rich behavioral and temporal window context.

Architecture:
  Raw Event JSON
        │
        ▼
  FlowStateManager (Temporal Sliding Window per Conversation Key)
        │
        ▼
  FeatureAggregator (Statistical Flow & Behavioral Feature Extraction)
        │
        ▼
  Exact 78-Feature DataFrame (Preserves ML_FEATURE_COLUMNS contract)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml.feature_aggregator import FeatureAggregator, ML_FEATURE_COLUMNS
from ml.flow_state import FlowStateManager, FlowWindow, NormalizedEvent

logger = logging.getLogger(__name__)


class TelemetryAdapter:
    """
    Stateful telemetry adapter bridging live network telemetry to trained ML models.
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        max_events_per_key: int = 200,
        max_active_keys: int = 5000,
    ):
        self.state_manager = FlowStateManager(
            window_seconds=window_seconds,
            max_events_per_key=max_events_per_key,
            max_active_keys=max_active_keys,
        )
        self.aggregator = FeatureAggregator()

    def process_event(
        self,
        raw_event: Dict[str, Any],
        stateful: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a single raw telemetry event through the behavioral window layer
        and compute the corresponding 78-feature ML row.

        Args:
            raw_event: Dictionary representing the TelemetryEvent.
            stateful: If True (default), updates and uses the rolling conversation window.
                      If False, treats the event in standalone isolation.

        Returns:
            Tuple of:
              - DataFrame with 1 row and exactly 78 columns (ML_FEATURE_COLUMNS).
              - Context dict with window metadata (window_event_count, duration_span_s, etc.).
        """
        if stateful:
            norm_event, window, host_ctx = self.state_manager.record_and_get_window(raw_event)
        else:
            norm_event = self.state_manager.parse_event(raw_event)
            window = FlowWindow(
                key=(norm_event.source_ip, norm_event.destination_ip, norm_event.destination_port),
                window_seconds=self.state_manager.window_seconds,
            )
            window.add_event(norm_event)
            host_ctx = {
                "host_attempts_in_window": 1,
                "window_event_count": 1,
                "window_span_seconds": norm_event.duration,
            }

        feature_row = self.aggregator.aggregate(
            current_event=norm_event,
            window=window,
            host_context=host_ctx,
        )

        df = pd.DataFrame([feature_row], columns=ML_FEATURE_COLUMNS)

        context = {
            "source_ip": norm_event.source_ip,
            "destination_ip": norm_event.destination_ip,
            "destination_port": norm_event.destination_port,
            "protocol": norm_event.protocol,
            "window_event_count": window.event_count,
            "window_span_s": round(window.duration_span_seconds, 3),
            "host_attempts": host_ctx.get("host_attempts_in_window", 1),
        }

        return df, context

    def event_to_dataframe(
        self,
        raw_event: Dict[str, Any],
        stateful: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience method returning just the 78-feature DataFrame.
        """
        df, _ = self.process_event(raw_event, stateful=stateful)
        return df

    def event_to_row(
        self,
        raw_event: Dict[str, Any],
        stateful: bool = True,
    ) -> Dict[str, float]:
        """
        Convenience method returning the 78-feature dictionary.
        """
        df, _ = self.process_event(raw_event, stateful=stateful)
        return df.iloc[0].to_dict()

    def batch_to_dataframe(
        self,
        events: List[Dict[str, Any]],
        stateful: bool = True,
    ) -> pd.DataFrame:
        """
        Process a list/batch of raw telemetry events in sequential order.
        """
        if not events:
            return pd.DataFrame(columns=ML_FEATURE_COLUMNS)

        rows = []
        for ev in events:
            df, _ = self.process_event(ev, stateful=stateful)
            rows.append(df.iloc[0])

        return pd.DataFrame(rows, columns=ML_FEATURE_COLUMNS)

    def reset(self) -> None:
        """Clear all active conversation windows and temporal state."""
        self.state_manager.reset()

    def get_summary(self, source_ip: str, destination_ip: str, destination_port: int) -> Dict[str, Any]:
        """Query state summary for an active conversation key."""
        return self.state_manager.get_summary(source_ip, destination_ip, destination_port)
