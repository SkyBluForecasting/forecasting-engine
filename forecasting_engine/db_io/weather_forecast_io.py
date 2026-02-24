from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence

import pandas as pd
from sqlalchemy import and_, exists, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from forecasting_db.models import WeatherForecast, WeatherVariable
from .base_io import BaseIO


class WeatherForecastIO(BaseIO):
    """Handles loading and writing weather forecast data."""

    def __init__(self, session: Session):
        super().__init__(session)

    def get_latest_issue_time(
        self, weather_site_id: str, variable: WeatherVariable
    ) -> Optional[datetime]:
        """Get the latest issue_time for a given weather site and variable."""
        return (
            self.session.query(func.max(WeatherForecast.issue_time))
            .filter(
                WeatherForecast.weather_site_id == weather_site_id,
                WeatherForecast.variable == variable,
            )
            .scalar()
        )

    def get_latest_issue_times(
        self, weather_site_id: str
    ) -> Dict[WeatherVariable, Optional[datetime]]:
        """
        Get latest issue_time per variable for a site (single query).

        Returns a dict like {WeatherVariable.temp_air_c: datetime|None, ...}.
        Variables with no data will not appear in the result.
        """
        rows = (
            self.session.query(
                WeatherForecast.variable,
                func.max(WeatherForecast.issue_time),
            )
            .filter(WeatherForecast.weather_site_id == weather_site_id)
            .group_by(WeatherForecast.variable)
            .all()
        )
        return {var: ts for var, ts in rows}

    def forecasts_exist_for_range(
        self,
        weather_site_id: str,
        start_time: datetime,
        end_time: datetime,
        variable: WeatherVariable,
        issue_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if any forecasts exist for the given target timestamp range.

        Optionally filter to a specific issue_time.
        """
        conds = [
            WeatherForecast.weather_site_id == weather_site_id,
            WeatherForecast.variable == variable,
            WeatherForecast.timestamp >= start_time,
            WeatherForecast.timestamp <= end_time,
        ]
        if issue_time is not None:
            conds.append(WeatherForecast.issue_time == issue_time)

        return bool(self.session.query(exists().where(and_(*conds))).scalar())

    def bulk_insert_forecasts(self, forecasts: List[dict]) -> int:
        """
        Bulk insert weather forecast data (idempotent).

        Expects dict keys: weather_site_id, timestamp, variable, value, issue_time
        Uses uq_weather_forecast_site_var_target_issue to ignore duplicates.
        """
        if not forecasts:
            return 0

        stmt = pg_insert(WeatherForecast).values(forecasts)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["weather_site_id", "variable", "timestamp", "issue_time"]
        )
        result = self.session.execute(stmt)
        self.session.commit()

        return int(result.rowcount or 0)

    def get_forecasts_for_site(
        self,
        weather_site_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[Sequence[WeatherVariable]] = None,
        issue_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Retrieve weather forecasts for a site as DataFrame.

        Returns columns: timestamp, variable, value, issue_time, weather_site_id
        """
        query = self.session.query(WeatherForecast).filter(
            WeatherForecast.weather_site_id == weather_site_id
        )

        if start_time is not None:
            query = query.filter(WeatherForecast.timestamp >= start_time)
        if end_time is not None:
            query = query.filter(WeatherForecast.timestamp <= end_time)
        if variables:
            query = query.filter(WeatherForecast.variable.in_(list(variables)))
        if issue_time is not None:
            query = query.filter(WeatherForecast.issue_time == issue_time)

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "variable": r.variable.value,
                    "value": r.value,
                    "issue_time": r.issue_time,
                    "weather_site_id": r.weather_site_id,
                }
                for r in rows
            ]
        )

    def from_df(self, df: pd.DataFrame, *args, **kwargs) -> int:
        """Write weather forecasts from DataFrame to DB (idempotent)."""
        required_cols = {
            "timestamp",
            "variable",
            "value",
            "weather_site_id",
            "issue_time",
        }
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        df_copy = df[list(required_cols)].copy()

        # Convert variable strings to enum if needed
        df_copy["variable"] = df_copy["variable"].apply(
            lambda x: WeatherVariable(x) if isinstance(x, str) else x
        )

        # Ensure timestamps are datetime (pandas can keep tz-aware)
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], utc=True)
        df_copy["issue_time"] = pd.to_datetime(df_copy["issue_time"], utc=True)

        records = df_copy.to_dict(orient="records")
        return self.bulk_insert_forecasts(records)

    def to_df(self) -> pd.DataFrame:
        """Load all weather forecasts into a pandas DataFrame."""
        rows = self.session.query(WeatherForecast).all()
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "weather_site_id": r.weather_site_id,
                    "timestamp": r.timestamp,
                    "variable": r.variable.value,
                    "value": r.value,
                    "issue_time": r.issue_time,
                }
                for r in rows
            ]
        )
