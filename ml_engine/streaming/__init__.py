try:
    from ml_engine.streaming.telemetry_adapter import TelemetryAdapter, ML_FEATURE_COLUMNS
    from ml_engine.streaming.flow_state import FlowStateManager, FlowWindow, NormalizedEvent
    from ml_engine.streaming.feature_aggregator import FeatureAggregator
except ImportError:
    try:
        from .telemetry_adapter import TelemetryAdapter, ML_FEATURE_COLUMNS
        from .flow_state import FlowStateManager, FlowWindow, NormalizedEvent
        from .feature_aggregator import FeatureAggregator
    except ImportError:
        from ml.telemetry_adapter import TelemetryAdapter, ML_FEATURE_COLUMNS
        from ml.flow_state import FlowStateManager, FlowWindow, NormalizedEvent
        from ml.feature_aggregator import FeatureAggregator

