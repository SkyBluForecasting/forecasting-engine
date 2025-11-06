import pandas as pd
import contextlib
import io
from typing import Optional
from forecasting_engine.openstef.pipeline.train_model import train_model_pipeline
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)
from forecasting_engine.db_io.measurement_io import MeasurementsIO
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.config import MLFLOW_TRACKING_URI

logger = get_logger(__name__)

# Hardcoded defaults for training
RES_MIN = 60
HORIZON_MIN = 8 * 24 * 60  # 8 days in minutes
QUANTILES = [10, 30, 50, 70, 90]
MODEL_NAME = "xgb"
FORECAST_TYPE = "demand"
LAT = 52.0
LON = 5.0


class TrainingManager:
    """Encapsulates the training workflow for assets."""

    def __init__(self, session):
        self.session = session
        self.measurements_io = MeasurementsIO(session)
        self.assets_io = AssetsIO(session)

    # --- Data preparation ---
    def _prepare_train_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw measurements into format expected by OpenSTEF."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.rename(columns={"load_kW": "load", "temp_C": "temp"})
        df = df.set_index("timestamp").sort_index()
        df["load"] = pd.to_numeric(df["load"], errors="coerce")
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        return df

    def _slice_train_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove the last horizon window to create training data."""
        test_len = int(HORIZON_MIN // RES_MIN)
        if len(df) <= test_len:
            raise ValueError(
                f"Not enough rows ({len(df)}) for test window ({test_len})"
            )
        return df.iloc[:-test_len]

    def _build_prediction_job(
        self, asset_id: str, model_id: Optional[str] = None
    ) -> PredictionJobDataClass:
        return PredictionJobDataClass(
            id=(model_id or asset_id),
            model=MODEL_NAME,
            quantiles=QUANTILES,
            forecast_type=FORECAST_TYPE,
            lat=LAT,
            lon=LON,
            horizon_minutes=HORIZON_MIN,
            resolution_minutes=RES_MIN,
            name=f"Forecast_{asset_id}",
            hyper_params={},
            feature_names=None,
            default_modelspecs=None,
            save_train_forecasts=True,
        )

    def train_asset(self, asset_id: str):
        df = self.measurements_io.to_df(asset_id)
        if df.empty:
            raise ValueError(f"No measurements found for asset {asset_id}")

        logger.info(f"Loaded {len(df)} measurement records for asset {asset_id}")

        train_df = self._prepare_train_df(df)
        train_df = self._slice_train_data(train_df)
        job = self._build_prediction_job(asset_id)

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                train_model_pipeline(
                    job,
                    train_df,
                    check_old_model_age=False,
                    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                    artifact_folder=None,
                )
        except LookupError as e:
            logger.error(f"No model found in MLflow for asset {asset_id}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Model training failed for asset {asset_id}: {e}")
            raise

        logger.info(f"Training successful for asset: {asset_id}")

    def train_all_assets(self):
        assets = self.assets_io.list_assets()
        total = len(assets)

        success_count = 0
        failure_count = 0
        failed_assets = []

        for i, asset in enumerate(assets, start=1):
            logger.info(
                f"[{i}/{total}] Training asset {asset.asset_uuid} ...", flush=True
            )
            try:
                self.train_asset(asset.asset_uuid)
                success_count += 1
            except Exception as e:
                failure_count += 1
                failed_assets.append(asset.asset_uuid)
                logger.exception(f"Training failed for asset {asset.asset_uuid}: {e}")

        logger.info(
            f"Training complete. "
            f"✅ Successful: {success_count}/{total}, "
            f"❌ Failed: {failure_count}/{total}"
        )

        if failed_assets:
            logger.warning(f"Failed assets: {', '.join(failed_assets)}")
