from forecasting_db.models import Measurement
from .base_io import BaseIO
import pandas as pd


class MeasurementsIO(BaseIO):
    """Handles loading measurement data from the database."""

    def to_df(self, asset_uuid: str, metrics: list[str] = None) -> pd.DataFrame:
        """
        Load measurements for an asset, pivoted by metric name.

        Args:
            asset_uuid: The asset UUID.
            metrics: Optional list of metric names to filter (e.g. ['load_kw', 'temp_c']).

        Returns:
            DataFrame with columns ['timestamp', 'load', 'temp', ...].
        """
        try:
            query = self.session.query(Measurement).filter(
                Measurement.asset_uuid == asset_uuid
            )
            if metrics:
                query = query.filter(Measurement.metric.in_(metrics))

            rows = query.all()
            if not rows:
                raise ValueError(f"No measurements found for asset {asset_uuid}")

            df = pd.DataFrame(
                [
                    {"timestamp": r.timestamp, "metric": r.metric, "value": r.value}
                    for r in rows
                ]
            )

            # Pivot so each metric becomes a column (e.g. load_kw, temp_c)
            df = df.pivot(
                index="timestamp", columns="metric", values="value"
            ).reset_index()
            df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            raise ValueError(f"Failed to load measurements for asset {asset_uuid}: {e}")

    def from_df(self, df: pd.DataFrame, asset_uuid: str):
        """
        Bulk insert measurement data from a DataFrame.

        Args:
            df: DataFrame with columns ['timestamp', 'metric', 'value'] or pivoted metrics.
            asset_uuid: The asset UUID.
        """
        if "metric" in df.columns:
            # Long form (timestamp, metric, value)
            df_to_insert = df.copy()
        else:
            # Wide form (timestamp, load_kw, temp_c, etc.)
            df_to_insert = df.melt(
                id_vars=["timestamp"], var_name="metric", value_name="value"
            )

        df_to_insert["asset_uuid"] = asset_uuid

        records = df_to_insert.to_dict(orient="records")
        self.session.bulk_insert_mappings(Measurement, records)
        self.session.commit()

    def resample_to_frequency(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample measurements to the given frequency defined in the Prediction Job.
        Handles uneven intervals and missing timestamps.
        """
        if df.empty:
            return df

        try:
            # Ensure proper datetime index
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" not in df.columns:
                    raise ValueError(
                        "DataFrame must have either a DatetimeIndex or 'timestamp' column"
                    )
                df = df.set_index("timestamp")

            # Resample and interpolate if needed
            df_resampled = df.resample(freq).mean().interpolate(limit_direction="both")

            # Optional sanity cleanup
            df_resampled = df_resampled.dropna(how="all")
            return df_resampled

        except Exception as e:
            raise ValueError(f"Failed to resample data to frequency {freq}: {e}")
