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
from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.config import MLFLOW_TRACKING_URI
from forecasting_engine.db_io.forecast_io import ForecastIO
from forecasting_engine.db_io.constraint_io import ConstraintsIO
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.db_io.measurement_io import MeasurementsIO
from forecasting_engine.db_io.forecast_run_io import ForecastRunIO
from forecasting_engine.db_io.prediction_job_io import PredictionJobIO
from forecasting_engine.shared.forecast_utils import (
    ForecastDataProcessor,
    normalize_forecast_columns,
)
from forecasting_engine.openstef.model.serializer import MLflowSerializer

logger = get_logger(__name__)


def run_asset_forecast(asset_id: str) -> dict:
    """Wrapper to run forecast for an asset with error handling."""
    try:
        return _run_asset_forecast_inner(asset_id)
    except FileNotFoundError as e:
        logger.error(f"[NOT FOUND] {e}")
        return {"asset_id": asset_id, "status": "not_found", "message": str(e)}
    except ValueError as e:
        logger.error(f"[BAD INPUT] {e}")
        return {"asset_id": asset_id, "status": "bad_input", "message": str(e)}
    except Exception as e:
        logger.exception(f"[FATAL] Unexpected error for asset {asset_id}: {e}")
        return {"asset_id": asset_id, "status": "fatal_error", "message": str(e)}


def _run_asset_forecast_inner(asset_id: str) -> dict:
    """Core runner: opens session, instantiates manager, generates forecast."""
    with SessionLocal() as session:
        fm = ForecastManager(session)
        fm.generate_forecast(asset_id)
    return {"asset_id": asset_id, "status": "success", "message": None}


# ============================
# Forecast Manager
# ============================


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

        measurements_df = self._load_measurements(asset_id)
        pj = self._build_prediction_job(asset_id)
        prepared_df = self._prepare_data(measurements_df, pj)
        forecast_df, mlflow_run_id = self._run_forecast(pj, prepared_df)
        self._save_forecast_results(forecast_df, pj, mlflow_run_id, asset_id)

        total_elapsed = time.time() - start_time
        logger.info(
            f"Total forecast generation time for asset {asset_id}: {total_elapsed:.2f}s"
        )
        return forecast_df

    def _load_measurements(self, asset_id: str):
        df = self.measurements_io.to_df(asset_id)
        if df.empty:
            raise ValueError(f"No measurements found for asset {asset_id}")
        logger.info(f"Loaded {len(df)} measurement records for asset {asset_id}")
        return df

    def _build_prediction_job(self, asset_id: str) -> PredictionJobDataClass:
        pj = PredictionJobDataClass(
            id=asset_id,
            model="xgb",
            quantiles=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            forecast_type="demand",
            lat=52.0,
            lon=5.0,
            horizon_minutes=48 * 60,
            resolution_minutes=60,
            name=asset_id,
            hyper_params={},
            feature_names=None,
            default_modelspecs=None,
            save_train_forecasts=True,
        )
        logger.info(
            f"Built prediction job for asset {asset_id} with {pj.horizon_minutes/60:.1f}h horizon"
        )
        return pj

    def _prepare_data(self, measurements_df, pj):
        processor = ForecastDataProcessor(measurements_df, pj)
        prepared_df = processor.add_forecast_horizon_nans()
        logger.info(f"Prepared asset {pj.id} data, total points: {len(prepared_df)}")
        return prepared_df

    def _run_forecast(self, pj, input_data):
        """Runs OpenSTEF forecast, suppressing stdout/stderr for MLflow."""
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            serializer = MLflowSerializer(self.mlflow_uri)
            latest_model_record = serializer._find_models(str(pj.id), max_results=1)
            if latest_model_record.empty:
                raise LookupError(f"No model found in MLflow for asset {pj.id}")

            latest_model = latest_model_record.iloc[0]
            mlflow_run_id = latest_model.run_id
            logger.info(
                f"Found MLflow model for asset {pj.id}, run_id: {mlflow_run_id}"
            )

            model, model_specs = serializer.load_model(experiment_name=str(pj.id))
            logger.info(f"Loaded model for asset {pj.id}")

            forecast_df = create_forecast_pipeline_core(
                pj, input_data, model, model_specs
            )
            forecast_df = normalize_forecast_columns(forecast_df)
            logger.info(
                f"Forecast generated for asset {pj.id}, {len(forecast_df)} data points"
            )

            return forecast_df, mlflow_run_id

    def _save_forecast_results(self, forecast_df, pj, mlflow_run_id, asset_id):
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
        ConstraintsIO(self.session).from_forecast(
            forecast_df, forecast_run_id, asset_id
        )
