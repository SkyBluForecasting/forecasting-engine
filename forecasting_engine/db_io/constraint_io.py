from forecasting_db.models import Constraint, Asset
from .base_io import BaseIO
from forecasting_engine.shared.logger_factory import get_logger
import pandas as pd

logger = get_logger(__name__)


class ConstraintsIO(BaseIO):
    """Handles saving and loading asset constraint violations."""

    def from_forecast(
        self,
        forecast_df: pd.DataFrame,
        forecast_run_id: str,
        asset_uuid: str,
    ):
        """
        Save constraint violations for an asset based on its forecast.

        Args:
            forecast_df: DataFrame with forecast values (must include 'timestamp' and 'forecast').
            forecast_run_id: ForecastRun ID associated with this forecast.
            asset_uuid: Asset ID to check against.
        """
        try:
            if forecast_df.empty:
                logger.warning(
                    f"Empty forecast DataFrame for constraint check, asset {asset_uuid}"
                )
                return

            # Validate required columns
            if "forecast" not in forecast_df.columns:
                raise ValueError(
                    f"Missing 'forecast' column for constraint check, asset {asset_uuid}"
                )

            # Load asset capacity
            asset = (
                self.session.query(Asset)
                .filter(Asset.asset_uuid == asset_uuid)
                .one_or_none()
            )
            if not asset:
                logger.warning(f"Asset {asset_uuid} not found in database")
                return

            if asset.capacity_kw is None:
                logger.debug(
                    f"Asset {asset_uuid} has no capacity rating, skipping constraint check"
                )
                return

            # Find violations: forecast > capacity
            violations = forecast_df[forecast_df["forecast"] > asset.capacity_kw].copy()
            if violations.empty:
                logger.debug(f"No constraint violations found for asset {asset_uuid}")
                return

            # Prepare DB rows
            db_rows = []
            for _, row in violations.iterrows():
                db_rows.append(
                    {
                        "asset_uuid": asset_uuid,
                        "timestamp": row["timestamp"],
                        "forecast_run_id": forecast_run_id,
                        "constraint_kw": row["forecast"] - asset.capacity_kw,
                    }
                )

            # Bulk insert
            self.session.bulk_insert_mappings(Constraint, db_rows)
            self.session.commit()
            logger.info(
                f"Saved {len(db_rows)} constraint violations for asset {asset_uuid}"
            )

        except Exception as e:
            logger.error(
                f"Failed to save constraint violations for asset {asset_uuid}: {e}"
            )
            self.session.rollback()
            raise

    def to_df(self, asset_uuid: str = None) -> pd.DataFrame:
        """Load constraint violations as DataFrame, optionally filtered by asset."""
        query = self.session.query(Constraint)
        if asset_uuid:
            query = query.filter(Constraint.asset_uuid == asset_uuid)
        rows = query.all()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(
            [
                {
                    "asset_uuid": r.asset_uuid,
                    "timestamp": r.timestamp,
                    "forecast_run_id": r.forecast_run_id,
                    "constraint_kw": r.constraint_kw,
                }
                for r in rows
            ]
        )
        return df

    def from_df(self, df: pd.DataFrame, *args, **kwargs):
        raise NotImplementedError("Use `from_forecast` instead of `from_df`")
