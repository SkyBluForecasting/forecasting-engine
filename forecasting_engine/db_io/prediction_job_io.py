from forecasting_engine.db_io.base_io import BaseIO
from forecasting_db.models import PredictionJob
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)
from forecasting_engine.orchestration.logger_factory import get_logger
import pandas as pd

logger = get_logger(__name__)


class PredictionJobIO(BaseIO):
    """DB I/O for PredictionJob table."""

    def get_or_create(self, pj: PredictionJobDataClass) -> int:
        """Get existing PredictionJob ID or create a new row."""
        try:
            # QUERY BY asset_uuid (string), not by integer primary key
            obj = (
                self.session.query(PredictionJob)
                .filter(PredictionJob.asset_uuid == pj.id)
                .first()
            )
            if obj:
                logger.debug(
                    f"Found existing PredictionJob ID {obj.id} for asset {pj.id}"
                )
                return obj.id

            new_obj = PredictionJob(
                asset_uuid=pj.id,
                model=pj.model,
                horizon_minutes=pj.horizon_minutes,
                frequency_minutes=pj.resolution_minutes,
                config={
                    "quantiles": pj.quantiles,
                    "forecast_type": pj.forecast_type,
                    "model_kwargs": getattr(pj, "model_kwargs", {}),
                },
            )
            self.session.add(new_obj)
            self.session.commit()
            logger.info(f"Created new PredictionJob ID {new_obj.id} for asset {pj.id}")
            return new_obj.id

        except Exception as e:
            logger.error(f"Failed to get/create PredictionJob for asset {pj.id}: {e}")
            self.session.rollback()
            raise

    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        """Optional: load PredictionJobs as DataFrame."""
        jobs = self.session.query(PredictionJob).all()
        return pd.DataFrame([j.__dict__ for j in jobs])

    def from_df(self, df: pd.DataFrame, *args, **kwargs):
        # Not needed for ForecastRunIO
        pass
