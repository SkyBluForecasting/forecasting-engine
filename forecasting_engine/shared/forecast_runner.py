"""Forecast runner: generates forecasts for assets, saves to DB."""

import time
import contextlib
import io
import pandas as pd
from datetime import datetime, timezone, timedelta

from dataclasses import dataclass

from forecasting_engine.openstef.pipeline.create_forecast import (
    create_forecast_pipeline_core,
)
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)
from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.config import (
    DEFAULT_FORECAST_HORIZON_MINUTES,
    MLFLOW_TRACKING_URI,
)
from forecasting_engine.db_io.assets_io import AssetsIO
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


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    prediction_job: PredictionJobDataClass
    model_run_id: str | None


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
        self.assets_io = AssetsIO(session)

    def generate_forecast(self, asset_id: str):
        """Generate forecast for an asset.
        - If asset is a leaf node and measured=True, runs model-based forecast
        - If asset has children, runs hierarchical forecast by summing child forecasts
        - Saves the resulting forecast to the database
        - Logs timing and status
        - Returns the generated forecast dataframe.
        """
        start_time = time.time()
        logger.info(f"Starting forecast generation for asset: {asset_id}")
        asset = self.assets_io.get_asset(asset_id)

        # ----- LEAF NODE (MODEL-BASED FORECASTS) ------
        if not asset.children:
            if not asset.measured:
                logger.info(
                    f"Skipping forecast generation for leaf asset {asset_id}: measured=False"
                )
                return None
            result = self._run_model_based_forecast(asset)
        # ----- NON-LEAF NODE (AGGREGATION FORECASTS) ----
        else:
            result = self._run_hierarchy_based_forecast(asset)

        if result is None:
            return None

        self._save_forecast_results(
            result.forecast_df,
            result.prediction_job,
            result.model_run_id,
            asset_id,
        )

        logger.info(
            f"Finished forecast for asset {asset_id} "
            f"in {time.time() - start_time:.2f}s"
        )
        return result.forecast_df

    def _load_measurements(self, asset_id: str):
        """
        Load historical measurements for a given asset from the database.

        Raises:
            ValueError if no measurements are found.

        Returns:
            DataFrame of measurements with timestamps and values.
        """
        df = self.measurements_io.to_df(asset_id)
        if df.empty:
            raise ValueError(f"No measurements found for asset {asset_id}")

        return df

    def _build_prediction_job(self, asset_id: str) -> PredictionJobDataClass:
        """
        Create a PredictionJobDataClass instance for the asset.
        """
        pj = PredictionJobDataClass(
            id=asset_id,
            model="xgb",
            quantiles=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            forecast_type="demand",
            lat=52.0,
            lon=5.0,
            horizon_minutes=DEFAULT_FORECAST_HORIZON_MINUTES,
            resolution_minutes=60,
            name=asset_id,
            hyper_params={},
            feature_names=None,
            default_modelspecs=None,
            save_train_forecasts=True,
        )
        return pj

    def _prepare_data(self, measurements_df, pj):
        """
        Prepare measurements for forecasting.

        - Adds NaNs for future timestamps based on forecast horizon
        - Any additional preprocessing required by ForecastDataProcessor

        Returns:
            Prepared DataFrame suitable for forecasting
        """
        processor = ForecastDataProcessor(measurements_df, pj)
        prepared_df = processor.add_forecast_horizon_nans()
        logger.info(f"Prepared asset {pj.id} data, total points: {len(prepared_df)}")
        return prepared_df

    def _get_target_horizon(self) -> tuple[datetime, datetime]:
        """
        Determine the target forecast horizon.

        Returns a tuple of (start_datetime, end_datetime) using the current time and default forecast horizon.
        """
        now = datetime.now(timezone.utc)
        horizon = timedelta(minutes=DEFAULT_FORECAST_HORIZON_MINUTES)
        return now, now + horizon

    def _fetch_latest_child_forecasts_in_timerange(
        self, asset, target_start: datetime, target_end: datetime
    ) -> list[pd.DataFrame] | None:
        """
        Fetch child forecasts for hierarchical aggregation.
        1) For each child, select the latest forecast that overlaps the target horizon.
            - If any child has no overlapping run, return None (no parent forecast).
        2) Compute the common time window that ALL children cover
        3) Return child forecasts ready for aggregation

        Returns: List of child dataframes, with columns ["timestamp", "<child_id>"] or None.
        """
        raw_child_dfs: list[tuple[str, pd.DataFrame]] = []

        for child in asset.children:
            df = self._get_latest_forecast_df(
                child.asset_uuid, target_start, target_end
            )
            if df is None or df.empty:
                logger.warning(
                    f"Child {child.asset_uuid} has no forecast overlapping target horizon. "
                    f"Skipping parent forecast for {asset.asset_uuid}"
                )
                return None
            raw_child_dfs.append((child.asset_uuid, df))

        # Window each child run covers (based on actual forecast points)
        child_mins = [df["timestamp"].min() for _, df in raw_child_dfs]
        child_maxs = [df["timestamp"].max() for _, df in raw_child_dfs]

        # Common window across children - extends backward to what all children cover
        updated_start = max(child_mins)
        updated_end = min(child_maxs)

        if updated_end <= updated_start:
            logger.warning(
                f"No common overlapping window across children for parent {asset.id}. "
                f"updated_start={updated_start}, updated_end={updated_end}"
            )
            return None

        # Slice + rename for aggregation
        sliced_child_dfs: list[pd.DataFrame] = []
        for child_id, df in raw_child_dfs:
            df2 = df[
                (df["timestamp"] >= updated_start) & (df["timestamp"] <= updated_end)
            ]
            df2 = df2[["timestamp", "forecast"]].rename(columns={"forecast": child_id})
            if df2.empty:
                logger.warning(
                    f"Child {child_id} has no points in common window. Skipping parent {asset.id}."
                )
                return None
            sliced_child_dfs.append(df2)

        return sliced_child_dfs

    def _get_latest_forecast_df(
        self, asset_id: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame | None:
        """
        Load the latest child forecast run that overlaps any part of the target horizon.

        Returns the full run DataFrame (not sliced to target horizon), because the
        hierarchical logic needs each child's full run coverage to compute a common window.
        """
        run_id = self.forecast_run_io.get_latest_overlapping_forecast_run_id(
            asset_uuid=asset_id,
            window_start=start_time,
            window_end=end_time,
        )
        if not run_id:
            return None

        df = self.forecast_io.to_df(run_id)
        if df is None or df.empty:
            return None

        return df.sort_values("timestamp").reset_index(drop=True)

    def _aggregate_child_forecasts(self, child_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate multiple child forecasts into a single parent forecast.

        - Merges DataFrames on timestamp (inner join)
        - Sums values across all children for each timestamp
        - Skips timestamps missing in any child
        Returns:
            Aggregated DataFrame with columns ['timestamp', 'forecast']
        """
        merged = child_dfs[0]
        for df in child_dfs[1:]:
            merged = merged.merge(df, on="timestamp", how="inner")

        if merged.empty:
            return pd.DataFrame(columns=["timestamp", "forecast"])

        value_cols = [c for c in merged.columns if c != "timestamp"]
        merged["forecast"] = merged[value_cols].sum(axis=1)
        return merged[["timestamp", "forecast"]]

    def _run_model_based_forecast(self, asset):
        """
        Run a model-based forecast for a leaf asset.

        - Loads measurements and prepares data
        - Finds the latest model in MLflow and loads it
        - Runs forecast pipeline
        - Normalizes forecast columns
        Returns:
            ForecastResult containing forecast DataFrame, prediction job, and MLflow run ID
        """
        measurements_df = self._load_measurements(asset.asset_uuid)
        pj = self._build_prediction_job(asset.asset_uuid)
        prepared_df = self._prepare_data(measurements_df, pj)

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
                pj, prepared_df, model, model_specs
            )
            forecast_df = normalize_forecast_columns(forecast_df)

            return ForecastResult(
                forecast_df=forecast_df, prediction_job=pj, model_run_id=mlflow_run_id
            )

    def _run_hierarchy_based_forecast(self, asset):
        """
        Run a hierarchical forecast for a non-leaf (parent) asset.

        - Defines target forecast horizon
        - Fetches latest child forecasts overlapping the horizon
        - Aggregates child forecasts into a parent forecast
        - Skips timestamps missing in any child
        Returns:
            ForecastResult with parent forecast, prediction job, and model_run_id=None
        """

        target_start, target_end = self._get_target_horizon()

        child_dfs = self._fetch_latest_child_forecasts_in_timerange(
            asset, target_start, target_end
        )
        if not child_dfs:
            return None

        parent_df = self._aggregate_child_forecasts(child_dfs)

        pj = self._build_prediction_job(asset.asset_uuid)

        return ForecastResult(
            forecast_df=parent_df, prediction_job=pj, model_run_id=None
        )

    def _save_forecast_results(self, forecast_df, pj, mlflow_run_id, asset_id):
        """
        Save forecast results to the database.

        - Creates/gets PredictionJob record
        - Creates ForecastRun record
        - Saves forecast data points
        - Saves constraints derived from forecast

        Args:
            forecast_df: DataFrame with timestamp/value columns
            pj: PredictionJobDataClass instance
            mlflow_run_id: MLflow run ID or 'hierarchical'
            asset_id: UUID of the asset
        """

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
