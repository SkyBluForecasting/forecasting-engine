from forecasting_engine.db_io.base_io import BaseIO
from forecasting_db.models import ForecastRun
from forecasting_engine.shared.logger_factory import get_logger
import pandas as pd
from sqlalchemy import func
import datetime
import uuid

logger = get_logger(__name__)


class ForecastRunIO(BaseIO):
    """DB I/O for ForecastRun table."""

    def create(
        self,
        prediction_job_id: int,
        model_run_id: str,
        asset_uuid: str,
        start_time: datetime.datetime = None,
        end_time: datetime.datetime = None,
        frequency_min: int = None,
    ) -> str:
        """
        Create a new ForecastRun.

        Args:
            prediction_job_id: ID of the prediction job.
            model_run_id: MLflow run ID.
            asset_uuid: ID of the asset.
            start_time: optional start timestamp.
            end_time: optional end timestamp.
            frequency_min: optional frequency in minutes.

        Returns:
            forecast_run_id (str)
        """
        try:
            new_run = ForecastRun(
                forecast_run_id=str(uuid.uuid4()),  # generate unique ID
                prediction_job_id=prediction_job_id,
                model_run_id=model_run_id,
                asset_uuid=asset_uuid,
                start_time=start_time,
                end_time=end_time,
                frequency_min=frequency_min,
            )
            self.session.add(new_run)
            self.session.commit()
            logger.info(
                f"Created ForecastRun {new_run.forecast_run_id} for asset {asset_uuid}"
            )
            return new_run.forecast_run_id

        except Exception as e:
            logger.error(f"Failed to create ForecastRun for asset {asset_uuid}: {e}")
            self.session.rollback()
            raise

    def update_status(self, forecast_run_id: str, **kwargs):
        """
        Update fields of a ForecastRun by forecast_run_id.
        """
        run = self.session.get(ForecastRun, forecast_run_id)
        if run:
            for key, value in kwargs.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            self.session.commit()

    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        runs = self.session.query(ForecastRun).all()
        df = pd.DataFrame([r.__dict__ for r in runs])

        return df

    def from_df(self, df: pd.DataFrame, *args, **kwargs):
        pass

    def get_latest_forecast_run_per_asset(
        self, asset_uuids: list[str]
    ) -> dict[str, datetime.datetime]:
        """
        Returns {asset_uuid: latest_forecast_start_time}
        """
        if not asset_uuids:
            return {}

        rows = (
            self.session.query(ForecastRun.asset_uuid, func.max(ForecastRun.start_time))
            .filter(ForecastRun.asset_uuid.in_(asset_uuids))
            .group_by(ForecastRun.asset_uuid)
            .all()
        )

        return {r[0]: r[1] for r in rows}
