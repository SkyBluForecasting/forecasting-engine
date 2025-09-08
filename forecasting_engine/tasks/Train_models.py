import argparse
import pandas as pd
from typing import List, Dict, Any, Optional

from forecasting_engine.openstef.pipeline.train_model import train_model_pipeline
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)

from forecasting_engine.orchestration.logger_factory import get_logger

from forecasting_engine.orchestration.s3_utils import (
    load_training_pd_from_s3,
    list_training_fsa_ids,
)

from forecasting_engine.config import MLFLOW_TRACKING_URI

logger = get_logger(__name__)

# Hardcode OpenSTEF job defaults for now
RES_MIN = int("60")  # 60-min in your notebook
HORIZON_MIN = float(8 * 24 * 60)  # 8 days ahead
QUANTILES = [10, 30, 50, 70, 90]  # match your notebook
MODEL_NAME = "xgb"
FORECAST_TYPE = "demand"
LAT = 52.0  # dummy location defaults
LON = 5.0


def split_train_test(df: pd.DataFrame, horizon_min: int, res_min: int, fsa_id: str):
    """
    Splits a DataFrame into training data by removing the last N rows for testing.

    Args:
        df: Full training DataFrame.
        horizon_min: Forecast horizon in minutes.
        res_min: Data resolution in minutes.
        fsa_id: Asset ID (used for error messages).

    Returns:
        train_data: DataFrame containing rows for training.
    """
    test_len = int(horizon_min // res_min)
    if len(df) <= test_len:
        raise ValueError(
            f"{fsa_id}: not enough rows ({len(df)}) for test window ({test_len})"
        )
    train_data = df.iloc[:-test_len, :]
    return train_data


def build_pj(fsa_id: str, model_id: Optional[str] = None) -> PredictionJobDataClass:
    """
    Build a PredictionJob similar to your notebook setup.
    """
    return PredictionJobDataClass(
        id=(model_id or fsa_id),
        model=MODEL_NAME,
        quantiles=QUANTILES,
        forecast_type=FORECAST_TYPE,
        lat=LAT,
        lon=LON,
        horizon_minutes=HORIZON_MIN,
        resolution_minutes=RES_MIN,
        name=f"Forecast_{fsa_id}",
        hyper_params={},  # supply tuned params here if you have them
        feature_names=None,  # let OpenSTEF choose defaults
        default_modelspecs=None,
        save_train_forecasts=True,  # match your notebook
    )


def train_single_fsa(fsa_id: str):
    """Train a model for a single FSA using an OpenSTEF pipeline"""

    try:
        df = load_training_pd_from_s3(asset_id=fsa_id)
    except FileNotFoundError as e:
        logger.error(f"Training data not found for asset {fsa_id}: {e}")
        raise
    except ValueError as e:
        logger.error(f"Training data for asset {fsa_id} is invalid: {e}")
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error loading training data for asset {fsa_id}: {e}"
        )
        raise

    train_data = split_train_test(
        df=df, horizon_min=HORIZON_MIN, res_min=RES_MIN, fsa_id=fsa_id
    )

    pj = build_pj(fsa_id)

    # Kick off OpenSTEF training with MLflow logging
    try:
        train_model_pipeline(
            pj,
            train_data,
            check_old_model_age=False,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            artifact_folder=None,  # Don't need to save write artifacts to disk, since they will already go to mlflow
        )

    except LookupError as e:
        logger.error(f"No model found in MLflow for asset {fsa_id}: {e}")
        raise
    except Exception as e:
        logger.exception(f"Model training failed for asset {fsa_id}: {e}")
        raise
    logger.info(f"Training successful for asset: {fsa_id}")


def train_all():
    keys = list_training_fsa_ids()
    results: List[Dict[str, Any]] = []
    for i, key in enumerate(keys, start=1):
        logger.info(f"[{i}/{len(keys)}] Training {key} ...", flush=True)
        results.append(train_single_fsa(key))
    return results


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(
        description="Train OpenSTEF models on EC2 using S3 training data + MLflow."
    )
    ap.add_argument(
        "--fsa-id",
        help="Train only this FSA (e.g., L9M). If omitted, trains all *_train.csv in the prefix.",
    )
    args = ap.parse_args()

    if args.fsa_id:
        train_single_fsa(args.fsa_id.upper())
    else:
        train_all()


if __name__ == "__main__":  # pragma: no cover
    main()
