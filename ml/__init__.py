"""
Compatibility layer redirecting legacy ml imports to ml_engine.streaming.
"""
from ml_engine.streaming import (
    TelemetryAdapter,
    ML_FEATURE_COLUMNS,
    FlowStateManager,
    FlowWindow,
    NormalizedEvent,
    FeatureAggregator,
)

__all__ = [
    "TelemetryAdapter",
    "ML_FEATURE_COLUMNS",
    "FlowStateManager",
    "FlowWindow",
    "NormalizedEvent",
    "FeatureAggregator",
]
