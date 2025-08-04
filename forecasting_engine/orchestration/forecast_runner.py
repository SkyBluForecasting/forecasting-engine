"""Pulls latest model from MLFlow, Pulls training data from S3, Calls openstef logic to generate forecasts, Pushes forecasts to S3"""

from forecasting_engine.openstef.pipeline.create_forecast import (
    create_forecast_pipeline,
)
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)
from forecasting_engine.orchestration.s3_utils import (
    load_training_pd_from_s3,
    save_forecast_csv_to_s3,
)

from forecasting_engine.orchestration.logger_factory import get_logger

from forecasting_engine.config import MLFLOW_TRACKING_URI

logger = get_logger(__name__)


def generate_forecast_for_asset(asset_id: str):
    """
    Loads the latest model for an asset and generates a forecast using OpenSTEF.

    Args:
        asset_id (str): Asset ID whose forecast is to be generated.

    Returns:
        pd.DataFrame: Forecast dataframe with predicted values.

    Raises:
        ValueError: If training data is malformed.
        FileNotFoundError: If training data is missing.
        LookupError: If no model is found in MLflow for the prediction job.
        Exception: For any other unforeseen failure during the forecast pipeline.
    """
    logger.info(f"Starting forecast generation for asset: {asset_id}")

    try:
        training_data = load_training_pd_from_s3(asset_id=asset_id)
    except FileNotFoundError as e:
        logger.error(f"Training data not found for asset {asset_id}: {e}")
        raise
    except ValueError as e:
        logger.error(f"Training data for asset {asset_id} is invalid: {e}")
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error loading training data for asset {asset_id}: {e}"
        )
        raise

    try:

        # Eventually pj will likely be an input. Hardcode for now.
        pj = PredictionJobDataClass(
            id=asset_id,
            model="xgb",
            quantiles=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            forecast_type="demand",
            lat=52.0,
            lon=5.0,
            horizon_minutes=48 * 60,  # 48 hour forecast
            resolution_minutes=60,  # hourly steps
            name=asset_id,
            hyper_params={},
            feature_names=None,
            default_modelspecs=None,
            save_train_forecasts=True,
        )

        forecast = create_forecast_pipeline(
            pj=pj,
            input_data=training_data,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        )

    except LookupError as e:
        logger.error(f"No model found in MLflow for asset {asset_id}: {e}")
        raise
    except Exception as e:
        logger.exception(f"Forecast pipeline failed for asset {asset_id}: {e}")
        raise

    logger.info(f"Forecast generation successful for asset: {asset_id}")

    save_forecast_csv_to_s3(df=forecast, asset_id=asset_id)

    return forecast
