# KRAMPS — PS14 AI Cyber Defence Platform

An intelligent, multi-stage cyber defense and threat detection platform engineered for mission-critical and tactical enterprise networks.

---

## 🚀 Overview

The **PS14 AI Cyber Defence Platform** continuously ingests network telemetry, extracts behavioral flow dynamics via rolling-window state tracking, performs dual-model machine learning inference (Supervised XGBoost + Unsupervised Isolation Forest), fuses detection scores, and tracks attack progressions aligned with the MITRE ATT&CK framework.

```
+-------------------------------------------------------------------------+
|                         Network Telemetry Stream                        |
|             (Live Simulation / Kafka Topic / Direct Ingestion)          |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  Stateful Flow Window & Normalization                   |
|       (Sliding Window Aggregation -> 78 CIC-IDS2017 Feature Vector)      |
+-------------------------------------------------------------------------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
+-----------------------------------+ +-----------------------------------+
|     Supervised Attack Detector    | |    Unsupervised Anomaly Model     |
|          (Trained XGBoost)        | |     (Trained Isolation Forest)    |
+-----------------------------------+ +-----------------------------------+
                  |                                   |
                  +-----------------+-----------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        Dynamic Score Fusion                             |
|          Combined Score = (0.6 * Supervised) + (0.4 * Anomaly)          |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                Threat Correlation & MITRE Progression                   |
|       (Heuristic ETA, Stage Tracker, SHAP Explanations & Alerts)        |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                     FastAPI Real-Time Control Plane                     |
|           (SSE Live Stream, Swagger Docs, Threat Dashboard API)         |
+-------------------------------------------------------------------------+
```

---

## 📂 Project Structure

The repository is organized into distinct, modular subsystems:

```
SIH/
├── backend/                       # FastAPI server, SSE stream, REST APIs
│   ├── app/
│   │   ├── core/                  # Core configurations & Kafka utilities
│   │   ├── telemetry/             # Telemetry schemas, consumers & storage
│   │   └── main.py                # Server app lifecycle & SimulationManager
│   └── __init__.py
│
├── simulator/                     # Network simulation and attack generation engine
│   ├── network/                   # Topology & assets (assets.json, topology.json)
│   ├── scenarios/                 # 8 Scenarios: ddos, brute_force, recon, c2, etc.
│   └── telemetry/                 # Event stream & continuous telemetry generator
│
├── ml_engine/                     # Unified ML Intelligence Core
│   ├── config/                    # config.yaml (model hyperparameters, weights)
│   ├── data/                      # Processed sample outputs and predictions
│   ├── evaluation/                # Model evaluation metrics & validation reports
│   ├── models/                    # Trained serialized model artifacts (.pkl, .json)
│   │   ├── supervised/            # XGBoost model & fitted preprocessor
│   │   └── anomaly/               # Isolation Forest model & normalization specs
│   ├── notebooks/                 # Model development & benchmark notebooks
│   ├── src/                       # Production ML Pipeline modules
│   │   ├── anomaly/               # Isolation Forest & SHAP explainer
│   │   ├── detection/             # XGBoost supervised classification
│   │   ├── features/              # Deterministic CIC-IDS2017 feature encoder
│   │   ├── fusion/                # Weighted score fusion engine
│   │   ├── ingestion/             # Telemetry loaders
│   │   ├── prediction/            # Alert correlator & MITRE attack progression
│   │   └── preprocessing/         # Raw data cleaning & sanitization
│   ├── streaming/                 # Real-time streaming ML adapter & live bridge
│   │   ├── feature_aggregator.py  # 78-feature behavioral window aggregator
│   │   ├── flow_state.py          # Bidirectional per-flow state tracker
│   │   ├── live_ml_bridge.py      # Kafka -> Window Adapter -> ML Inference bridge
│   │   └── telemetry_adapter.py   # TelemetryEvent -> CIC-IDS2017 transformer
│   └── main.py                    # Batch training & offline prediction CLI
│
├── experiments/                   # Exploratory classification benchmarks & data prep
│   ├── benchmarks/                # 12 baseline classifiers (SVM, KNN, MLP, etc.)
│   ├── data_prep/                 # Initial data cleaning, encoding & join scripts
│   └── docs/                      # Original field notes, Slurm scripts & checklists
│
├── ml/                            # Backwards-compatibility import layer
├── infrastructure/                # Docker compose orchestration (Kafka, Zookeeper)
├── tests/                         # Unit & schema test suite
├── run_server.py                  # One-click unified runner
├── test_e2e.py                    # Offline end-to-end simulation & ML test
├── requirements.txt               # Unified project dependencies
└── pytest.ini                     # Pytest configuration
```

---

## ⚡ Quick Start

### 1. Installation

Ensure Python 3.10+ is installed, then install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Unified Backend Server

Starts the FastAPI server with live simulated network traffic and real-time ML inference:

```bash
python run_server.py
```

Options:
- `--port 8000`: Change server port (default: `8000`).
- `--scenario ddos`: Force a specific scenario (`normal`, `ddos`, `brute_force`, `reconnaissance`, `lateral_movement`, `exfiltration`, `dns_c2`, `communication_disruption`).
- `--interval 0.2`: Interval in seconds between simulated events (default: `0.5s`).

Once running:
- **API Server**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Live Event Stream (SSE)**: [http://localhost:8000/api/stream](http://localhost:8000/api/stream)
- **Recent Telemetry**: [http://localhost:8000/api/telemetry/recent](http://localhost:8000/api/telemetry/recent)
- **Active Alerts**: [http://localhost:8000/api/alerts](http://localhost:8000/api/alerts)

### 3. Run the End-to-End Offline Test

Test the entire pipeline (Simulator -> Stateful Window -> XGBoost + Isolation Forest -> Alert Firing) without Kafka or Docker:

```bash
python test_e2e.py --events 30
```

Force a scenario:
```bash
python test_e2e.py --events 25 --scenario brute_force
python test_e2e.py --events 25 --scenario ddos
```

### 4. Train or Predict with ML Engine

```bash
# Run batch prediction on a telemetry file:
python ml_engine/main.py --mode predict --data path/to/telemetry.csv

# Train models on a new dataset:
python ml_engine/main.py --mode train --data path/to/dataset.csv
```

### 5. Run Automated Tests

```bash
pytest
```

---

## 🛡️ Cyber Threat Scenarios

The integrated simulation engine reproduces 8 distinct network traffic profiles:

| Scenario | Target / Behavior | Key Telemetry Characteristics |
| :--- | :--- | :--- |
| `normal` | Routine enterprise communications | HTTP/HTTPS browsing, DNS queries, balanced packet rates |
| `ddos` | High-volume volumetric flood | TCP/UDP port 443 flood, compressed intervals, high packet bursts |
| `brute_force` | Authentication brute force (SSH/RDP) | Rapid short-duration connection attempts to ports 22, 3389 |
| `reconnaissance`| Port scanning / host discovery | Probing wide ranges of destination ports with minimal bytes |
| `lateral_movement`| SMB / Internal pivoting | Traffic to port 445 / internal RPC traversing subnet boundaries |
| `exfiltration` | Data hoarding and outbound egress | Disproportionate `bytes_out` to external IP addresses |
| `dns_c2` | Command & Control via DNS | High-entropy subdomains queried against internal DNS servers |
| `communication_disruption` | Network gateway disruption | Gateway target congestion, elevated packet loss profiles |

---

## 🔒 Security & License

- Licensed under the terms specified in [`LICENSE`](file:///c:/Users/Krishna/Desktop/SIH/LICENSE).
- Built for defensive cybersecurity monitoring and telemetry analysis.
