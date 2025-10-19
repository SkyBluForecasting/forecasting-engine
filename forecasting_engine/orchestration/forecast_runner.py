"""Forecast runner: generates forecasts for assets, saves to DB."""

import time

import contextlib
import io

from forecasting_engine.openstef.pipeline.create_forecast import (
    create_forecast_pipeline_core,
)
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)
from forecasting_engine.orchestration.logger_factory import get_logger
from forecasting_engine.config import MLFLOW_TRACKING_URI
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.db_io.forecast_io import ForecastIO
from forecasting_engine.db_io.constraint_io import ConstraintsIO
from forecasting_engine.db_io.measurement_io import MeasurementsIO
from forecasting_engine.db_io.forecast_run_io import ForecastRunIO
from forecasting_engine.db_io.prediction_job_io import PredictionJobIO
from forecasting_engine.orchestration.forecast_utils import (
    ForecastDataProcessor,
    normalize_forecast_columns,
)
from forecasting_engine.openstef.model.serializer import MLflowSerializer

logger = get_logger(__name__)


def build_prediction_job(asset_id: str) -> PredictionJobDataClass:
    """Build a dummy prediction job for testing/forecast generation."""
    return PredictionJobDataClass(
        id=asset_id,
        model="xgb",
        quantiles=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        forecast_type="demand",
        lat=52.0,
        lon=5.0,
        horizon_minutes=48 * 60,
        resolution_minutes=60,  # hourly
        name=asset_id,
        hyper_params={},
        feature_names=None,
        default_modelspecs=None,
        save_train_forecasts=True,
    )


def run_openstef_forecast(pj, input_data, mlflow_tracking_uri):
    start_time = time.time()
    logger.info(f"Starting OpenSTEF forecast for asset {pj.id}")
    try:
        serializer = MLflowSerializer(mlflow_tracking_uri)

        latest_model_record = serializer._find_models(str(pj.id), max_results=1)
        if latest_model_record.empty:
            raise LookupError(f"No model found in MLflow for asset {pj.id}")
        latest_model = latest_model_record.iloc[0]
        mlflow_run_id = latest_model.run_id
        logger.info(f"Found MLflow model for asset {pj.id}, run_id: {mlflow_run_id}")

        model, model_specs = serializer.load_model(experiment_name=str(pj.id))
        logger.info(f"Loaded model for asset {pj.id}")

        forecast_df = create_forecast_pipeline_core(pj, input_data, model, model_specs)
        elapsed_time = time.time() - start_time
        logger.info(
            f"Forecast generated for asset {pj.id} in {elapsed_time:.2f}s, "
            f"{len(forecast_df)} data points"
        )
        return forecast_df, mlflow_run_id
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(
            f"OpenSTEF forecast failed for asset {pj.id} after {elapsed_time:.2f}s: {e}"
        )
        raise


class ForecastManager:
    """Encapsulates the workflow of generating and saving forecasts."""

    def __init__(self, session, mlflow_uri=MLFLOW_TRACKING_URI):
        self.session = session
        self.mlflow_uri = mlflow_uri
        self.pj_io = PredictionJobIO(session)
        self.forecast_run_io = ForecastRunIO(session)
        self.forecast_io = ForecastIO(session)
        self.measurements_io = MeasurementsIO(session)

    def generate_forecast(self, asset_id: str):
        start_time = time.time()
        logger.info(f"Starting forecast generation for asset: {asset_id}")

        # Load measurements
        measurements_df = self.measurements_io.to_df(asset_id)
        if measurements_df.empty:
            raise ValueError(f"No measurements found for asset {asset_id}")
        logger.info(
            f"Loaded {len(measurements_df)} measurement records for asset {asset_id}"
        )

        # Build prediction job
        pj = build_prediction_job(asset_id)
        logger.info(
            f"Built prediction job for asset {asset_id} with {pj.horizon_minutes/60:.1f}h horizon"
        )

        # Preprocess + add horizon NaNs
        processor = ForecastDataProcessor(measurements_df, pj)
        prepared_df = processor.add_forecast_horizon_nans()
        logger.info(f"Prepared asset {asset_id} data, total points: {len(prepared_df)}")

        # Suppress all MLflow/OpenSTEF stdout/print logs
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            forecast_df, mlflow_run_id = run_openstef_forecast(
                pj, prepared_df, self.mlflow_uri
            )

        # Normalize and save
        forecast_df = normalize_forecast_columns(forecast_df)

        pj_id = self.pj_io.get_or_create(pj)
        forecast_run_id = self.forecast_run_io.create(
            prediction_job_id=pj_id,
            model_run_id=mlflow_run_id,
            asset_uuid=asset_id,
            start_time=forecast_df["timestamp"].min().to_pydatetime(),
            end_time=forecast_df["timestamp"].max().to_pydatetime(),
            frequency_min=pj.resolution_minutes,
        )
        self.forecast_io.from_df(forecast_df, forecast_run_id)

        # Save constraint violations
        constraints_io = ConstraintsIO(self.session)
        constraints_io.from_forecast(
            forecast_df=forecast_df,
            forecast_run_id=forecast_run_id,
            asset_uuid=asset_id,
        )

        total_elapsed = time.time() - start_time
        logger.info(
            f"Total forecast generation time for asset {asset_id}: {total_elapsed:.2f}s"
        )
        return forecast_df


def generate_forecast_for_asset(asset_id: str):
    logger.info(f"=== Starting forecast generation for asset: {asset_id} ===")
    with SessionLocal() as session:
        manager = ForecastManager(session)
        return manager.generate_forecast(asset_id)
