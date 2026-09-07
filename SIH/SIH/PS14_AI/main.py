"""
PS14 AI Cyber Defence Platform — Main Pipeline Orchestrator.

Usage:
    python main.py --mode train   --data <path/to/telemetry.csv> [--config config/config.yaml]
    python main.py --mode predict --data <path/to/telemetry.csv> [--config config/config.yaml]

Modes:
    train:
        Loads raw telemetry, fits and saves separate preprocessing encoders for
        supervised and unsupervised baseline splits, trains the XGBoost detector
        and Isolation Forest anomaly model, and saves all artifacts.
        Does NOT evaluate model quality.

    predict:
        Loads raw telemetry, applies respective preprocessor, runs models,
        fuses scores, generates SHAP explanations, correlates alerts,
        and estimates attack progression. Saves output CSV and JSON.

IMPORTANT:
    Model quality must NOT be assessed here. Evaluation will be performed
    separately after training on the actual operational telemetry dataset.
"""

import argparse
import json
import logging
import os
import sys
import gc
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Module imports ────────────────────────────────────────────────────────────
from src.ingestion import TelemetryLoader
from src.preprocessing import TelemetryCleaner
from src.features import FeatureEncoder
from src.detection import XGBoostDetector
from src.anomaly import AnomalyDetector, SHAPExplainer
from src.fusion import ScoreFusion
from src.prediction import AlertCorrelator, AttackPredictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PS14")


# =============================================================================
# Configuration Loader
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load and validate YAML configuration."""
    abs_path = os.path.join(PROJECT_ROOT, config_path) if not os.path.isabs(config_path) else config_path
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Configuration file not found: {abs_path}\n"
            f"Expected at: config/config.yaml (relative to project root)."
        )
    with open(abs_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate mandatory sections.
    required_sections = ["pipeline", "paths", "preprocessing", "label", "model", "anomaly", "fusion"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(
            f"Configuration file is missing required sections: {missing}. "
            f"Check config/config.yaml."
        )

    # Validate fusion weights.
    fw = config["fusion"].get("detection_weight", 0)
    aw = config["fusion"].get("anomaly_weight", 0)
    if abs(fw + aw - 1.0) > 1e-6:
        raise ValueError(
            f"config.yaml: fusion.detection_weight ({fw}) + "
            f"fusion.anomaly_weight ({aw}) = {fw + aw:.4f} (must be 1.0)."
        )

    return config


# =============================================================================
# Path Helpers
# =============================================================================

def abs_path(relative_path: str) -> str:
    """Resolve a path relative to the project root."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


def resolve_model_paths(config: dict) -> dict:
    """Return absolute paths for all model and preprocessor artifacts."""
    sup_dir = abs_path(config["paths"]["supervised_model_dir"])
    ano_dir = abs_path(config["paths"]["anomaly_model_dir"])
    return {
        "supervised_dir":  sup_dir,
        "anomaly_dir":     ano_dir,
        "xgboost_model":   os.path.join(sup_dir, "xgboost_model.pkl"),
        "preprocessor":    os.path.join(sup_dir, "preprocessor.pkl"),
        "xgb_metadata":    os.path.join(sup_dir, "metadata.json"),
        "anomaly_model":   os.path.join(ano_dir, "anomaly_model.pkl"),
        "anomaly_preprocessor": os.path.join(ano_dir, "anomaly_preprocessor.pkl"),
        "anomaly_normalization": os.path.join(ano_dir, "normalization.json"),
        "anomaly_metadata": os.path.join(ano_dir, "metadata.json"),
    }


# =============================================================================
# Shared: Load and Preprocess Telemetry
# =============================================================================

def load_and_clean(data_path: str, config: dict) -> pd.DataFrame:
    """Load raw telemetry and return a cleaned DataFrame."""
    logger.info(f"[Pipeline] Loading telemetry from: {data_path}")
    loader = TelemetryLoader()
    raw_df = loader.load(data_path)

    pp_cfg = config["preprocessing"]
    cleaner = TelemetryCleaner(
        drop_columns=pp_cfg.get("drop_columns", []),
        timestamp_col=pp_cfg.get("timestamp_col") or "",
        max_nan_fraction=pp_cfg.get("max_nan_fraction", 0.5),
        numeric_impute_strategy=pp_cfg.get("numeric_impute_strategy", "median"),
        handle_infinity=pp_cfg.get("handle_infinity", True),
    )
    cleaned_df = cleaner.clean(raw_df)
    logger.info(f"[Pipeline] Cleaned: {len(cleaned_df):,} rows × {len(cleaned_df.columns)} columns.")
    return cleaned_df


# =============================================================================
# TRAIN MODE
# =============================================================================

def run_train(data_path: str, config: dict, paths: dict) -> None:
    """
    Full training pipeline.

    Steps:
      1. Load & clean telemetry.
      2. Split into train and validation sets FIRST (validation is NEVER seen by SMOTE/preprocessor fitting).
      3. Identify normal/benign baseline from train split for Isolation Forest.
      4. Fit supervised preprocessor only on train split, transform both splits.
      5. Train XGBoost on transformed train split (with SMOTE only on training data).
      6. Fit anomaly preprocessor on normal baseline, transform.
      7. Fit Isolation Forest on normal baseline, save normalization parameters.
    """
    print("=" * 80)
    print("  PS14 AI CYBER DEFENCE PLATFORM — TRAINING MODE")
    print("=" * 80)
    import gc

    # ── Step 1: Load and Clean ───────────────────────────────────────────
    cleaned_df = load_and_clean(data_path, config)

    pp_cfg = config["preprocessing"]
    label_cfg = config["label"]
    target_col = pp_cfg["target_col"]

    if target_col not in cleaned_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in training data.\n"
            f"Available columns: {list(cleaned_df.columns)}\n"
            f"Set preprocessing.target_col in config.yaml to the correct column name."
        )

    # ── Step 2: Separate X and y (Memory conscious) ─────────────────────
    logger.info("[Pipeline] Separating raw feature matrix and target column...")
    y_raw = cleaned_df[target_col].copy()
    X_raw = cleaned_df.drop(columns=[target_col])

    del cleaned_df
    gc.collect()

    # ── Step 3: Train / Validation Split (Validation set never seen) ─────
    logger.info("[Pipeline] Splitting data into train/validation splits...")
    from sklearn.model_selection import train_test_split
    strat = y_raw if len(y_raw.unique()) > 1 else None
    try:
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_raw, y_raw,
            test_size=config["model"].get("train_test_split", 0.2),
            random_state=config["model"]["xgboost"].get("random_state", 42),
            stratify=strat
        )
    except Exception as exc:
        logger.warning(f"Stratified split failed: {exc}. Performing standard split.")
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_raw, y_raw,
            test_size=config["model"].get("train_test_split", 0.2),
            random_state=config["model"]["xgboost"].get("random_state", 42)
        )

    del X_raw, y_raw
    gc.collect()

    # ── Step 4: Extract Normal Baseline for Isolation Forest ─────────────
    logger.info("[Pipeline] Extracting benign baseline from training split...")
    benign_values = [str(v).strip().lower() for v in label_cfg.get("benign_values", [])]

    def is_benign(val):
        return str(val).strip().lower() in benign_values

    benign_mask = y_train_raw.apply(is_benign)
    X_anomaly_train_raw = X_train_raw[benign_mask].copy()

    logger.info(
        f"[Pipeline] Training benign baseline samples: {len(X_anomaly_train_raw):,} "
        f"({benign_mask.mean():.1%} of training set)."
    )

    # ── Step 5: Fit and Transform Supervised Preprocessor ─────────────────
    print("\n[STAGE 1a] Fitting Supervised Preprocessor on Training Split...")
    encoder = FeatureEncoder(
        target_col=target_col,
        benign_values=label_cfg.get("benign_values", []),
        categorical_columns=pp_cfg.get("categorical_columns", []),
        hex_columns=pp_cfg.get("hex_columns", []),
        labels_already_binary=label_cfg.get("labels_already_binary", False),
    )
    encoder.fit(X_train_raw)

    X_train = encoder.transform(X_train_raw)
    y_train = encoder.encode_target(y_train_raw)

    X_val = encoder.transform(X_val_raw)
    y_val = encoder.encode_target(y_val_raw)

    # Save preprocessor artifact
    os.makedirs(paths["supervised_dir"], exist_ok=True)
    encoder.save(paths["preprocessor"])

    # Clean intermediate variables to save memory
    del X_train_raw, X_val_raw, y_train_raw, y_val_raw
    gc.collect()

    # ── Step 6: Train and Save XGBoost Detector ───────────────────────────
    print("\n[STAGE 1b] Training XGBoost Supervised Detector...")
    model_cfg = config["model"]
    xgb_cfg = model_cfg["xgboost"]
    detector = XGBoostDetector(
        **xgb_cfg,
        use_smote=model_cfg.get("use_smote", True),
        train_test_split_ratio=model_cfg.get("train_test_split", 0.2),
    )
    # Trains model (applying SMOTE inside training set only)
    detector.train(X_train, y_train)
    detector.save_model(paths["xgboost_model"])

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": detector.feature_names,
        "target_col": target_col,
        "n_features": len(detector.feature_names or []),
        "xgboost_params": xgb_cfg,
        "use_smote": model_cfg.get("use_smote", True),
        "train_test_split": model_cfg.get("train_test_split", 0.2),
        "decision_threshold": model_cfg.get("decision_threshold", 0.5),
        "n_train_samples": int(len(X_train)),
        "label_distribution": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
    }
    with open(paths["xgb_metadata"], "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[Pipeline] XGBoost metadata saved to: {paths['xgb_metadata']}")

    del X_train, y_train, X_val, y_val
    gc.collect()

    # ── Step 7: Fit and Transform Anomaly Preprocessor ───────────────────
    print("\n[STAGE 2a] Fitting Anomaly Preprocessor on Benign Baseline...")
    anomaly_encoder = FeatureEncoder(
        target_col=target_col,
        benign_values=label_cfg.get("benign_values", []),
        categorical_columns=pp_cfg.get("categorical_columns", []),
        hex_columns=pp_cfg.get("hex_columns", []),
        labels_already_binary=label_cfg.get("labels_already_binary", False),
    )
    anomaly_encoder.fit(X_anomaly_train_raw)
    X_anomaly_train = anomaly_encoder.transform(X_anomaly_train_raw)

    # Save anomaly preprocessor
    os.makedirs(paths["anomaly_dir"], exist_ok=True)
    anomaly_encoder.save(paths["anomaly_preprocessor"])

    del X_anomaly_train_raw
    gc.collect()

    # ── Step 8: Train and Save Anomaly Detector (Isolation Forest) ───────
    print("\n[STAGE 2b] Training Isolation Forest Anomaly Detector...")
    ano_cfg = config.get("anomaly", {})
    anomaly_detector = AnomalyDetector(
        contamination=ano_cfg.get("contamination", 0.05),
        n_estimators=ano_cfg.get("n_estimators", 100),
        random_state=ano_cfg.get("random_state", 42),
        max_samples=ano_cfg.get("max_samples", "auto"),
        train_on_benign_only=False, # Already subsetted to benign baseline
    )
    anomaly_detector.fit(X_anomaly_train)
    anomaly_detector.save_model(paths["anomaly_model"], paths["anomaly_normalization"])

    ano_metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "contamination": ano_cfg.get("contamination", 0.05),
        "n_estimators": ano_cfg.get("n_estimators", 100),
        "random_state": ano_cfg.get("random_state", 42),
        "score_min": anomaly_detector._score_min,
        "score_max": anomaly_detector._score_max,
        "n_train_samples": int(len(X_anomaly_train)),
    }
    with open(paths["anomaly_metadata"], "w") as f:
        json.dump(ano_metadata, f, indent=2)
    logger.info(f"[Pipeline] Anomaly metadata saved to: {paths['anomaly_metadata']}")

    del X_anomaly_train
    gc.collect()

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE. Artifacts saved:")
    print(f"    Supervised Preprocessor: {paths['preprocessor']}")
    print(f"    XGBoost Model          : {paths['xgboost_model']}")
    print(f"    Anomaly Preprocessor   : {paths['anomaly_preprocessor']}")
    print(f"    Anomaly Model          : {paths['anomaly_model']}")
    print(f"    Anomaly Normalization  : {paths['anomaly_normalization']}")
    print("=" * 80)
    print("\n  NOTE: Model quality has NOT been evaluated.")
    print("  Run evaluation separately on the real operational dataset after training.")


# =============================================================================
# PREDICT MODE
# =============================================================================

def run_predict(data_path: str, config: dict, paths: dict, output_dir: str) -> None:
    """
    Full inference pipeline.

    Steps:
      1. Load & clean telemetry.
      2. Load fitted preprocessors, transform features separately.
      3. Load XGBoost model, predict attack_probability.
      4. Load anomaly model, compute anomaly_score using saved normalization.
      5. Fuse scores (60/40 configurable).
      6. Generate SHAP explanations (optional, failure-isolated).
      7. Correlate alerts into sessions.
      8. Estimate heuristic attack progression per session.
      9. Save output CSV and JSON.
    """
    print("=" * 80)
    print("  PS14 AI CYBER DEFENCE PLATFORM — INFERENCE MODE")
    print("=" * 80)

    # Validate model artifacts exist before loading telemetry.
    for artifact_name, artifact_path in [
        ("XGBoost model", paths["xgboost_model"]),
        ("Supervised Preprocessor", paths["preprocessor"]),
        ("Anomaly model", paths["anomaly_model"]),
        ("Anomaly Preprocessor", paths["anomaly_preprocessor"]),
        ("Anomaly Normalization", paths["anomaly_normalization"]),
    ]:
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(
                f"{artifact_name} artifact not found: {artifact_path}\n"
                f"Run training first: python main.py --mode train --data <path>"
            )

    # ── Step 1: Load and Clean ────────────────────────────────────────────
    cleaned_df = load_and_clean(data_path, config)

    # ── Step 2: Supervised Inference ──────────────────────────────────────
    logger.info(f"[Pipeline] Loading supervised preprocessor from: {paths['preprocessor']}")
    encoder = FeatureEncoder.load(paths["preprocessor"])
    transformed_sup_df = encoder.transform(cleaned_df)
    X_sup = transformed_sup_df.drop(columns=[encoder.target_col], errors="ignore")

    print("\n[STAGE 1] XGBoost Attack Probability Prediction...")
    detector = XGBoostDetector()
    detector.load_model(paths["xgboost_model"])
    attack_probs = detector.predict_attack_probability(X_sup)

    # ── Step 3: Anomaly Inference ─────────────────────────────────────────
    logger.info(f"[Pipeline] Loading anomaly preprocessor from: {paths['anomaly_preprocessor']}")
    anomaly_encoder = FeatureEncoder.load(paths["anomaly_preprocessor"])
    transformed_ano_df = anomaly_encoder.transform(cleaned_df)
    X_ano = transformed_ano_df.drop(columns=[anomaly_encoder.target_col], errors="ignore")

    print("\n[STAGE 2] Anomaly Score Computation...")
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_model(paths["anomaly_model"], paths["anomaly_normalization"])
    anomaly_scores = anomaly_detector.predict_anomaly_score(X_ano)

    # ── Step 4: Score Fusion ───────────────────────────────────────────────
    fusion_cfg = config.get("fusion", {})
    fusion = ScoreFusion(
        detection_weight=fusion_cfg.get("detection_weight", 0.6),
        anomaly_weight=fusion_cfg.get("anomaly_weight", 0.4),
    )
    alert_scores = fusion.fuse(attack_probs, anomaly_scores)

    # Attach computed scores to the output DataFrame.
    output_df = cleaned_df.copy()
    output_df["attack_probability"] = np.round(attack_probs, 4)
    output_df["anomaly_score"] = np.round(anomaly_scores, 4)
    output_df["alert_score"] = np.round(alert_scores, 4)

    # ── Step 5: SHAP Explanations (XGBoost features) ──────────────────────
    shap_cfg = config.get("shap", {})
    if shap_cfg.get("enabled", True):
        print("\n[STAGE 2b] SHAP Feature Attribution Explanations...")
        explainer = SHAPExplainer(
            model=detector.model,
            top_k=shap_cfg.get("top_k", 3),
            background_samples=shap_cfg.get("background_samples", 100),
        )
        output_df["shap_top_features"] = explainer.explain(X_sup)
    else:
        output_df["shap_top_features"] = "disabled"

    # ── Step 6: Alert Correlation ─────────────────────────────────────────
    print("\n[STAGE 3] Alert Correlation...")
    corr_cfg = config.get("correlation", {})
    correlator = AlertCorrelator(
        alert_threshold=config["model"].get("decision_threshold", 0.5),
        attacker_ip_candidates=corr_cfg.get("attacker_ip_candidates", ["ip.src"]),
        target_ip_candidates=corr_cfg.get("target_ip_candidates", ["ip.dst"]),
        timestamp_fragments=corr_cfg.get("timestamp_fragments", []),
    )
    alert_sessions = correlator.correlate(output_df)
    logger.info(f"[Pipeline] Correlated into {len(alert_sessions)} attack sessions.")

    # ── Step 7: Attack Progression ─────────────────────────────────────────
    prog_cfg = config.get("progression", {})
    predictor = AttackPredictor(config=prog_cfg)
    session_predictions = predictor.predict_all_sessions(alert_sessions)

    # Display session summaries.
    print("\n" + "-" * 75)
    print("  STAGE 3 — ATTACK SESSION PROGRESSION SUMMARY (top 3)")
    print("-" * 75)
    for i, p in enumerate(session_predictions[:3]):
        print(f"\n  [SESSION {i+1}]  ID: {p['session_id']}")
        print(f"    ├─ Events         : {p['event_count']}")
        print(f"    ├─ Risk Index     : {p['risk']}")
        print(f"    ├─ Current Tactic : {p['current_tactic']}")
        print(f"    ├─ Next Tactic    : {p['next_tactic']}")
        print(f"    ├─ Target         : {p['target']}")
        print(f"    ├─ ETA            : {p['ETA']}")
        print(f"    └─ Intervention   : {p['recommended_intervention']}")

    # ── Step 8: Save Outputs ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    csv_path = os.path.join(output_dir, f"predictions_{timestamp_tag}.csv")
    output_df.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, f"sessions_{timestamp_tag}.json")
    with open(json_path, "w") as f:
        json.dump(session_predictions, f, indent=2)

    print("\n" + "=" * 80)
    print("  PREDICTION COMPLETE. Outputs saved:")
    print(f"    Predictions CSV : {csv_path}")
    print(f"    Sessions JSON   : {json_path}")
    print("=" * 80)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="PS14 AI Cyber Defence Pipeline",
        description="Train or run inference on network telemetry data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train:
    python main.py --mode train --data CIC_IDS2017_combined.csv

  Predict:
    python main.py --mode predict --data new_telemetry.csv

  Custom config:
    python main.py --mode train --data CIC_IDS2017_combined.csv --config config/config.yaml
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Execution mode: 'train' to fit models, 'predict' to run inference.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input telemetry CSV (or JSON/JSONL/Parquet) file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file. Default: config/config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for prediction artifacts. Defaults to paths.processed_data_dir from config.",
    )
    args = parser.parse_args()

    # Load configuration.
    config = load_config(args.config)
    logger.info(f"[Pipeline] Loaded: {config['pipeline']['name']} v{config['pipeline']['version']}")

    # Resolve artifact paths from config.
    paths = resolve_model_paths(config)
    output_dir = args.output_dir or abs_path(config["paths"]["processed_data_dir"])

    # Resolve data path (allow relative to project root).
    data_path = args.data if os.path.isabs(args.data) else os.path.join(PROJECT_ROOT, args.data)

    if args.mode == "train":
        run_train(data_path=data_path, config=config, paths=paths)
    elif args.mode == "predict":
        run_predict(data_path=data_path, config=config, paths=paths, output_dir=output_dir)


if __name__ == "__main__":
    main()
