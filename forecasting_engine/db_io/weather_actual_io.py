from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from forecasting_db.models import WeatherActual, WeatherVariable
from .base_io import BaseIO


class WeatherActualIO(BaseIO):
    """Handles loading and writing actual weather data."""

    def __init__(self, session: Session):
        super().__init__(session)

    def get_latest_reading_timestamp(
        self, weather_site_id: str, variable: WeatherVariable
    ) -> Optional[datetime]:
        """Get the latest timestamp for a given weather site and variable."""
        return (
            self.session.query(func.max(WeatherActual.timestamp))
            .filter(
                WeatherActual.weather_site_id == weather_site_id,
                WeatherActual.variable == variable,
            )
            .scalar()
        )

    def get_latest_reading_timestamps(
        self, weather_site_id: str
    ) -> Dict[WeatherVariable, Optional[datetime]]:
        """
        Get latest timestamp per variable for a site (single query).

        Returns a dict like {WeatherVariable.temp_air_c: datetime|None, ...}.
        Variables with no data will not appear in the result.
        """
        rows = (
            self.session.query(
                WeatherActual.variable,
                func.max(WeatherActual.timestamp),
            )
            .filter(WeatherActual.weather_site_id == weather_site_id)
            .group_by(WeatherActual.variable)
            .all()
        )
        return {var: ts for var, ts in rows}

    def bulk_insert_readings(self, readings: List[dict]) -> int:
        """
        Bulk insert weather actual readings (idempotent).

        Expects dict keys: weather_site_id, timestamp, variable, value
        Uses uq_weather_actual_site_var_ts to ignore duplicates.
        """
        if not readings:
            return 0

        stmt = pg_insert(WeatherActual).values(readings)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["weather_site_id", "variable", "timestamp"]
        )

        result = self.session.execute(stmt)
        self.session.commit()

        # rowcount is number of rows actually inserted for ON CONFLICT DO NOTHING
        return int(result.rowcount or 0)

    def get_readings_for_site(
        self,
        weather_site_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[Sequence[WeatherVariable]] = None,
    ) -> pd.DataFrame:
        """
        Retrieve weather actual readings for a site as DataFrame.

        Returns columns: timestamp, variable, value, weather_site_id
        """
        query = self.session.query(WeatherActual).filter(
            WeatherActual.weather_site_id == weather_site_id
        )

        if start_time is not None:
            query = query.filter(WeatherActual.timestamp >= start_time)
        if end_time is not None:
            query = query.filter(WeatherActual.timestamp <= end_time)
        if variables:
            query = query.filter(WeatherActual.variable.in_(list(variables)))

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "variable": r.variable.value,
                    "value": r.value,
                    "weather_site_id": r.weather_site_id,
                }
                for r in rows
            ]
        )

    def from_df(self, df: pd.DataFrame, *args, **kwargs) -> int:
        """Write weather actuals from DataFrame to DB (idempotent)."""
        required_cols = {"timestamp", "variable", "value", "weather_site_id"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        df_copy = df[list(required_cols)].copy()

        # Convert variable strings to enum if needed
        df_copy["variable"] = df_copy["variable"].apply(
            lambda x: WeatherVariable(x) if isinstance(x, str) else x
        )

        # Ensure timestamps are datetime (pandas can keep tz-aware)
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], utc=True)

        records = df_copy.to_dict(orient="records")
        return self.bulk_insert_readings(records)

    def to_df(self) -> pd.DataFrame:
        """Load all weather actuals into a pandas DataFrame."""
        rows = self.session.query(WeatherActual).all()
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "weather_site_id": r.weather_site_id,
                    "timestamp": r.timestamp,
                    "variable": r.variable.value,
                    "value": r.value,
                }
                for r in rows
            ]
        )
