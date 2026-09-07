"""Legacy compatibility shim for ml.live_ml_bridge."""
from ml_engine.streaming.live_ml_bridge import *

if __name__ == "__main__":
    from ml_engine.streaming.live_ml_bridge import main
    main()
