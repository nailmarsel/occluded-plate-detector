#!/usr/bin/env python3
"""
Register a model version in MLflow Model Registry.

Usage examples:

    # Register car detector v1.2.0
    python scripts/register_models.py car_detector \
        --model-path models/car_detector.pt \
        --version 1.2.0 \
        --metrics '{"precision": 0.92, "recall": 0.88}' \
        --dataset cars_v3

    # Register OCR model
    python scripts/register_models.py ocr \
        --model-path models/ocr_model.onnx \
        --version 2.1.0 \
        --metrics '{"accuracy": 0.85}'

    # Promote an existing version to Production
    python scripts/register_models.py car_detector --promote-to production --version 1.2.0
"""
import argparse
import json
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Configuration – adjust according to your environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAMES = ["car_detector", "plate_detector", "ocr", "embedding"]
ARTIFACT_DIR = "artifacts"  # local temp dir used for logging

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def log_and_register(
    model_name: str,
    model_path: str,
    version: str,
    metrics: dict | None = None,
    dataset: str | None = None,
    description: str | None = None,
) -> None:
    """
    Log a model artifact, metrics, and register a new version in MLflow.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with mlflow.start_run(run_name=f"{model_name}_v{version}") as run:
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("version", version)
        if dataset:
            mlflow.log_param("dataset", dataset)
        if description:
            mlflow.set_tag("mlflow.note.content", description)

        # Log metrics
        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

        # Log model artifact (weights)
        mlflow.log_artifact(model_path, artifact_path=ARTIFACT_DIR)

        # Register the model in the registry
        model_uri = f"runs:/{run.info.run_id}/{ARTIFACT_DIR}"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        # Add version description and tags
        client.update_model_version(
            name=model_name,
            version=registered_model.version,
            description=f"{description or ''} Trained on {dataset or 'N/A'}.",
        )
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="dataset",
            value=dataset or "unknown",
        )

        print(
            f"✅ Model '{model_name}' version {registered_model.version} "
            f"registered successfully."
        )
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Metrics: {metrics or 'none'}")


def promote_model(model_name: str, version: str, stage: str) -> None:
    """
    Transition a model version to the specified stage (Staging / Production / Archived).
    """
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    print(f"🚀 Model '{model_name}' version {version} moved to '{stage}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Register / manage models in MLflow Registry"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- register subcommand ---
    register_parser = subparsers.add_parser("register", help="Register a new model version")
    register_parser.add_argument(
        "model",
        choices=MODEL_NAMES,
        help="Model identifier",
    )
    register_parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the model file (e.g., models/car_detector.pt)",
    )
    register_parser.add_argument(
        "--version",
        required=True,
        help="Semantic version of the model (e.g., 1.2.0)",
    )
    register_parser.add_argument(
        "--metrics",
        type=json.loads,
        help='JSON string with metrics, e.g. \'{"accuracy": 0.9}\'',
    )
    register_parser.add_argument(
        "--dataset",
        help="Identifier of the training dataset (e.g., cars_v3)",
    )
    register_parser.add_argument(
        "--description",
        help="Optional description of the model version",
    )

    # --- promote subcommand ---
    promote_parser = subparsers.add_parser("promote", help="Promote a model version to a stage")
    promote_parser.add_argument("model", choices=MODEL_NAMES)
    promote_parser.add_argument("--version", required=True)
    promote_parser.add_argument(
        "--stage",
        required=True,
        choices=["Staging", "Production", "Archived"],
        help="Target stage",
    )

    args = parser.parse_args()

    if args.command == "register":
        log_and_register(
            model_name=args.model,
            model_path=args.model_path,
            version=args.version,
            metrics=args.metrics,
            dataset=args.dataset,
            description=args.description,
        )
    elif args.command == "promote":
        promote_model(args.model, args.version, args.stage)


if __name__ == "__main__":
    main()