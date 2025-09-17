#!/usr/bin/env python3
import os
import re
import io
import argparse
from typing import List, Dict, Any, Optional
from datetime import timezone

import boto3
import botocore
import pandas as pd

from openstef.pipeline.train_model import train_model_pipeline
from openstef.data_classes.prediction_job import PredictionJobDataClass

# ====== CONFIG (override via environment variables if you like) ======
S3_BUCKET = os.getenv("TRAINING_BUCKET", "forecasting-forecasts")
S3_PREFIX = os.getenv("TRAINING_PREFIX", "training/")  # expects <FSA>_train.csv under here

# CSV schema: accept either `datetime` or `timestamp` for the time column; `load` must exist
TIME_COL_CANDIDATES = ["datetime", "timestamp", "index"]
LOAD_COL = os.getenv("LOAD_COL", "load")

# OpenSTEF job defaults
RES_MIN = int(os.getenv("RES_MIN", "60"))            # 60-min in your notebook
HORIZON_MIN = int(os.getenv("HORIZON_MIN", str(8*24*60)))  # 8 days ahead
QUANTILES = [10, 30, 50, 70, 90]                     # match your notebook
MODEL_NAME = os.getenv("MODEL_NAME", "xgb")
FORECAST_TYPE = os.getenv("FORECAST_TYPE", "demand")
LAT = float(os.getenv("LAT", "52.0"))                # dummy location defaults
LON = float(os.getenv("LON", "5.0"))

# MLflow (you can also set these as env vars; see section below)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")         # e.g., "file:/home/ubuntu/mlruns" or "http://mlflow:5000"
MLFLOW_ARTIFACT_FOLDER = os.getenv("MLFLOW_ARTIFACT_FOLDER")   # e.g., "s3://your-mlflow-artifacts/" or local dir
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "openstef-forecasting")

# ====================================================================

s3 = boto3.client("s3")


def _list_training_csv_keys(bucket: str, prefix: str) -> List[str]:
    """List keys like <FSA>_train.csv under S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    pat = re.compile(r".+_train\.csv$", re.IGNORECASE)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if pat.match(os.path.basename(k)):
                keys.append(k)
    keys.sort()
    return keys


def _read_s3_csv_to_df(bucket: str, key: str) -> pd.DataFrame:
    """Load CSV from S3 to DataFrame; normalize time column to UTC index."""
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    df = pd.read_csv(buf)

    # Resolve time column
    time_col = None
    for cand in TIME_COL_CANDIDATES:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise ValueError(
            f"{key}: needs a time column named one of {TIME_COL_CANDIDATES}."
        )
    if LOAD_COL not in df.columns:
        raise ValueError(f"{key}: missing required '{LOAD_COL}' column.")

    # Standardize to pandas DateTimeIndex in UTC
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    if df[time_col].isna().any():
        raise ValueError(f"{key}: invalid timestamps in '{time_col}'.")

    df = df.sort_values(time_col).set_index(time_col)
    # Keep at least these two; any extra columns (e.g., 'temp') are fine
    # Columns order isn't critical for train_model_pipeline
    return df


def _fsa_from_key(key: str) -> str:
    base = os.path.basename(key)
    m = re.match(r"([A-Za-z0-9]+)_train\.csv$", base)
    if not m:
        raise ValueError(f"Cannot derive FSA from key: {key}")
    return m.group(1).upper()


def _build_pj(fsa_id: str, model_id: Optional[str] = None) -> PredictionJobDataClass:
    """
    Build a PredictionJob similar to your notebook setup.
    """
    pj_dict = dict(
        id=(model_id or fsa_id),
        model=MODEL_NAME,
        quantiles=QUANTILES,
        forecast_type=FORECAST_TYPE,
        lat=LAT,
        lon=LON,
        horizon_minutes=HORIZON_MIN,
        resolution_minutes=RES_MIN,
        name=f"Forecast_{fsa_id}",
        hyper_params={},           # supply tuned params here if you have them
        feature_names=None,        # let OpenSTEF choose defaults
        default_modelspecs=None,
        save_train_forecasts=True, # match your notebook
    )
    return PredictionJobDataClass(**pj_dict)


def _train_single_fsa(bucket: str, key: str) -> Dict[str, Any]:
    fsa_id = _fsa_from_key(key)
    df = _read_s3_csv_to_df(bucket, key)

    # Split last N samples for test (same idea as your notebook: ~8 days for 60-min data)
    test_len = int((HORIZON_MIN // RES_MIN) * 1)  # same horizon length for test window
    if len(df) <= test_len:
        raise ValueError(f"{key}: not enough rows ({len(df)}) for test window ({test_len}).")

    train_data = df.iloc[:-test_len, :]

    pj = _build_pj(fsa_id)

    # Kick off OpenSTEF training with MLflow logging
    fitted_model, report, modelspecs_out, datasets = train_model_pipeline(
        pj,
        train_data,
        check_old_model_age=False,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
        artifact_folder=MLFLOW_ARTIFACT_FOLDER,   # MLflow "artifact location" (root for this run's artifacts)
    )

    # Try to extract MLflow run info if present in report
    run_id = None
    try:
        rdict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        run_id = (rdict.get("mlflow") or {}).get("run_id")
    except Exception:
        pass

    return {
        "fsa_id": fsa_id,
        "s3_key": key,
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "artifact_folder": MLFLOW_ARTIFACT_FOLDER,
            "run_id": run_id,
        },
    }


def train_all() -> List[Dict[str, Any]]:
    keys = _list_training_csv_keys(S3_BUCKET, S3_PREFIX)
    results: List[Dict[str, Any]] = []
    for i, key in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}] Training {key} ...", flush=True)
        results.append(_train_single_fsa(S3_BUCKET, key))
    return results


def train_one(fsa_id: str) -> Dict[str, Any]:
    key = f"{S3_PREFIX}{fsa_id}_train.csv"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
    except botocore.exceptions.ClientError as e:
        raise FileNotFoundError(f"Not found: s3://{S3_BUCKET}/{key}") from e
    print(f"Training {key} ...", flush=True)
    return _train_single_fsa(S3_BUCKET, key)


def main():
    ap = argparse.ArgumentParser(description="Train OpenSTEF models on EC2 using S3 training data + MLflow.")
    ap.add_argument("--fsa-id", help="Train only this FSA (e.g., L9M). If omitted, trains all *_train.csv in the prefix.")
    args = ap.parse_args()

    if args.fsa_id:
        out = train_one(args.fsa_id.upper())
        print(pd.Series(out).to_string())
    else:
        out = train_all()
        df = pd.DataFrame(out)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
