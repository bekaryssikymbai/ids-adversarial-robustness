"""
Benchmark script for adversarial robustness evaluation.
Runs a test pipeline on sample data to verify reproducibility.
"""

import argparse
import logging
import os
import sys
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "benchmark.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(path: str, target_col: str, drop_cols: list) -> tuple:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Keep only numeric features for sample test
    X = X.select_dtypes(include=[np.number])
    return X, y


def train_baseline(X_train, y_train, config: dict, model_name: str):
    if model_name == "xgboost":
        model = xgb.XGBClassifier(**config["models"]["xgboost"])
    elif model_name == "random_forest":
        model = RandomForestClassifier(**config["models"]["random_forest"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model


def generate_fgsm_attack(model, X, y, epsilon: float) -> pd.DataFrame:
    """Simple FGSM-style perturbation using surrogate gradients."""
    X_adv = X.copy().values
    # Placeholder: random perturbation scaled by epsilon
    # In full pipeline, gradients come from surrogate model
    noise = np.random.uniform(-epsilon, epsilon, X_adv.shape)
    X_adv = X_adv + noise
    return pd.DataFrame(X_adv, columns=X.columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["output"]["log_dir"])

    logging.info("Starting benchmark pipeline")
    logging.info(f"Using sample data: {args.sample}")

    data_path = config["data"]["sample_path"] if args.sample else config["data"]["raw_path"]

    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        sys.exit(1)

    X, y = load_data(
        data_path,
        config["data"]["target_column"],
        config["data"]["drop_columns"],
    )

    logging.info(f"Loaded data: {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["seed"],
        stratify=y,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for model_name in ["xgboost", "random_forest"]:
        logging.info(f"Training baseline: {model_name}")
        model = train_baseline(X_train_scaled, y_train, config, model_name)

        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)

        logging.info(f"{model_name} — F1: {f1:.4f}, Accuracy: {acc:.4f}")

        results.append({
            "model": model_name,
            "attack": "none",
            "epsilon": 0.0,
            "f1_score": f1,
            "accuracy": acc,
        })

        # Run FGSM attack at different epsilon
        for eps in config["attacks"]["fgsm"]["epsilon"]:
            X_adv = generate_fgsm_attack(model, X_test_scaled, y_test, eps)
            y_adv_pred = model.predict(X_adv)
            f1_adv = f1_score(y_test, y_adv_pred, average="weighted")
            acc_adv = accuracy_score(y_test, y_adv_pred)

            logging.info(f"{model_name} | FGSM ε={eps} — F1: {f1_adv:.4f}, ΔF1: {f1 - f1_adv:.4f}")

            results.append({
                "model": model_name,
                "attack": "fgsm",
                "epsilon": eps,
                "f1_score": f1_adv,
                "accuracy": acc_adv,
                "delta_f1": f1 - f1_adv,
            })

    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    output_path = os.path.join(
        config["output"]["results_dir"],
        "sample_benchmark.csv" if args.sample else "full_benchmark.csv",
    )
    pd.DataFrame(results).to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")
    logging.info("Benchmark pipeline finished successfully")


if __name__ == "__main__":
    main()
