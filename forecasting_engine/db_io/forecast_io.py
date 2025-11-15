from .base_io import BaseIO
from forecasting_db.models import Forecast
from forecasting_engine.shared.logger_factory import get_logger
import pandas as pd

logger = get_logger(__name__)


class ForecastIO(BaseIO):
    """Handles saving and loading forecasts."""

    REQUIRED_COLUMNS = ["timestamp", "forecast"]

    def from_df(self, forecast_df: pd.DataFrame, forecast_run_id: int):
        """Save a forecast DataFrame to the database (DBC style: forecast_value + lower/upper)."""
        try:
            if forecast_df.empty:
                raise ValueError(
                    f"Empty forecast DataFrame for run_id {forecast_run_id}"
                )

            # Validate required columns
            missing_cols = [
                col for col in self.REQUIRED_COLUMNS if col not in forecast_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns for run_id {forecast_run_id}: {missing_cols}"
                )

            # Map DataFrame columns to DB model fields
            db_mappings = []
            for _, row in forecast_df.iterrows():
                mapping = {
                    "forecast_run_id": forecast_run_id,
                    "forecast_value": row["forecast"],
                    "timestamp": row["timestamp"],
                    "lower_q": row.get("p05"),  # map lower quantile
                    "upper_q": row.get("p95"),  # map upper quantile
                    "description": row.get("description"),
                }
                db_mappings.append(mapping)

            # Bulk insert
            self.session.bulk_insert_mappings(Forecast, db_mappings)
            self.session.commit()
            logger.info(
                f"Saved {len(db_mappings)} forecast records for run_id {forecast_run_id}"
            )

        except Exception as e:
            logger.error(f"Failed to save forecast for run_id {forecast_run_id}: {e}")
            self.session.rollback()
            raise

    def to_df(self, forecast_run_id: str) -> pd.DataFrame:
        """Load forecasts for a given run ID into a DataFrame."""
        try:
            rows = (
                self.session.query(Forecast)
                .filter(Forecast.forecast_run_id == forecast_run_id)
                .all()
            )
            if not rows:
                logger.warning(
                    f"No forecast records found for run_id {forecast_run_id}"
                )
                return pd.DataFrame()

            df = pd.DataFrame(
                [
                    {
                        "timestamp": r.timestamp,
                        "forecast": r.forecast_value,
                        "lower_q": r.lower_q,
                        "upper_q": r.upper_q,
                        "description": getattr(r, "description", None),
                    }
                    for r in rows
                ]
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            logger.info(
                f"Loaded {len(df)} forecast records for run_id {forecast_run_id}"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to load forecasts for run_id {forecast_run_id}: {e}")
            raise
