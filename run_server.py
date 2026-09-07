"""
PS14 AI Cyber Defence - Unified Server & Live Dashboard Runner
=============================================================
Starts the FastAPI backend server with live telemetry simulation
and real-time ML normalization & inference.

Usage:
    python run_server.py
    python run_server.py --port 8000 --auto-start --scenario ddos
    python run_server.py --interval 0.2
"""

import os
import sys
import argparse
import uvicorn
import requests
import time
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def auto_start_loop(port: int, scenario: str = None, interval: float = 0.5):
    """Wait for server to boot, then trigger simulation start."""
    time.sleep(2.0)
    url = f"http://127.0.0.1:{port}/api/simulator/start"
    params = {"interval": interval}
    if scenario:
        params["scenario"] = scenario
    try:
        res = requests.post(url, params=params)
        if res.status_code == 200:
            print(f"[Runner] Live Simulation auto-started! Mode: {scenario or 'RANDOM_MIX'} (every {interval}s)")
    except Exception as e:
        print(f"[Runner] Note: Auto-start ping: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="PS14 AI Cyber Defence - Unified Server Runner"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host IP (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    parser.add_argument("--scenario", default=None, choices=[
        "normal", "ddos", "reconnaissance", "brute_force",
        "lateral_movement", "exfiltration", "dns_c2", "communication_disruption"
    ], help="Force a single scenario or leave None for random mix")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between simulated events (default 0.5s)")

    args = parser.parse_args()

    from backend.app.main import sim_manager
    sim_manager.forced_scenario = args.scenario
    sim_manager.interval = args.interval

    print()
    print("=" * 75)
    print("  PS14 AI CYBER DEFENCE — UNIFIED BACKEND SERVER")
    print("=" * 75)
    print(f"  API Server URL  : http://localhost:{args.port}")
    print(f"  API Docs (Swagger): http://localhost:{args.port}/docs")
    print(f"  Live Stream SSE : http://localhost:{args.port}/api/stream")
    print(f"  Recent Events   : http://localhost:{args.port}/api/telemetry/recent")
    print(f"  Alerts Endpoint : http://localhost:{args.port}/api/alerts")
    print(f"  Simulation Mode : {args.scenario.upper() if args.scenario else 'RANDOM WEIGHTED MIX'} (every {args.interval}s)")
    print("=" * 75)
    print()

    uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=False, log_level="warning")

if __name__ == "__main__":
    main()
