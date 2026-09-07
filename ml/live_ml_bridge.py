"""Legacy compatibility shim for ml.live_ml_bridge."""
from ml_engine.streaming.live_ml_bridge import *

if __name__ == "__main__":
    from ml_engine.streaming.live_ml_bridge import main
    # When executed directly as python -m ml.live_ml_bridge
    import sys
    from ml_engine.streaming.live_ml_bridge import run_live_bridge, parser
    args = parser.parse_args()
    run_live_bridge(ml_root=args.ml_root, kafka_servers=args.kafka, window_seconds=args.window)
