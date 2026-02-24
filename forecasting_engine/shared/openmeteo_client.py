"""Open-Meteo weather API client for fetching weather forecasts and historical data."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from forecasting_engine.shared.logger_factory import get_logger

logger = get_logger(__name__)


class OpenMeteoClient:
    """Client for fetching weather data from Open-Meteo API."""

    BASE_URL = "https://api.open-meteo.com/v1"
    FORECAST_URL = f"{BASE_URL}/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Map Open-Meteo variables to database WeatherVariable names (strings)
    VARIABLE_MAPPING: Dict[str, str] = {
        "temperature_2m": "temp_air_c",
        "wind_speed_10m": "wind_speed_ms",
        "shortwave_radiation": "ghi_wm2",
    }

    def __init__(
        self, timeout: int = 30, max_retries: int = 3, backoff_factor: float = 0.5
    ):
        """
        Initialize the OpenMeteo client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Number of retries for transient errors (429/5xx)
            backoff_factor: Exponential backoff factor for retries
        """
        self.timeout = timeout
        self.session = requests.Session()

        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,  # we'll call raise_for_status() ourselves for better logging
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @staticmethod
    def _validate_coords(latitude: float, longitude: float) -> None:
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"latitude must be between -90 and 90. Got {latitude}")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"longitude must be between -180 and 180. Got {longitude}")

    def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Log with URL + body snippet for debugging
            body = resp.text
            if body and len(body) > 500:
                body = body[:500] + "…"
            logger.error(
                "HTTP Error %s for %s", resp.status_code, getattr(resp, "url", url)
            )
            logger.error("Response body: %s", body)
            raise e
        return resp.json()

    def fetch_forecast(
        self, latitude: float, longitude: float, hours: int = 60
    ) -> pd.DataFrame:
        """
        Fetch hourly weather forecast for the next N hours.

        Returns:
            DataFrame with columns: timestamp, variable, value, issue_time
        """
        self._validate_coords(latitude, longitude)

        now = datetime.now(timezone.utc)

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(self.VARIABLE_MAPPING.keys()),
            "timezone": "UTC",
            "forecast_hours": hours,
        }

        try:
            data = self._get_json(self.FORECAST_URL, params=params)
            return self._parse_hourly_data(data, is_forecast=True, issue_time=now)

        except requests.HTTPError:
            # _get_json already logged details
            return pd.DataFrame()
        except Exception as e:
            logger.exception(
                "Failed to fetch forecast for lat=%s lon=%s: %s", latitude, longitude, e
            )
            return pd.DataFrame()

    def fetch_historical(
        self,
        latitude: float,
        longitude: float,
        end_date: datetime,
        start_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical weather data.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            end_date: End datetime for historical data. Results are filtered to only include timestamps up to end_date.
            start_date: Start date for historical data. If None, defaults to 30 days before end_date.
                       Accepts datetime or date; only date portion is used for API call.
        """
        self._validate_coords(latitude, longitude)

        try:
            end_dt = end_date
            end_d: date = end_dt.date()

            # Determine start date
            if start_date is None:
                start_d: date = end_d - timedelta(days=30)
            else:
                start_d = start_date.date() if isinstance(start_date, datetime) else start_date  # type: ignore[assignment]

            if start_d > end_d:
                logger.warning("start_date (%s) is after end_date (%s)", start_d, end_d)
                return pd.DataFrame()

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "hourly": ",".join(self.VARIABLE_MAPPING.keys()),
                "timezone": "UTC",
            }

            data = self._get_json(self.ARCHIVE_URL, params=params)
            df = self._parse_hourly_data(data, is_forecast=False)

            # Filter out any data that is past the end time
            if not df.empty:
                df = df[df["timestamp"] <= end_dt]

            return df

        except requests.HTTPError:
            return pd.DataFrame()
        except Exception as e:
            logger.exception(
                "Failed to fetch historical data for lat=%s lon=%s: %s",
                latitude,
                longitude,
                e,
            )
            return pd.DataFrame()

    def _parse_hourly_data(
        self,
        data: Dict[str, Any],
        is_forecast: bool = True,
        issue_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Vectorized parsing of hourly data from Open-Meteo response.

        Returns:
            DataFrame with columns: timestamp, variable, value[, issue_time]
        """
        hourly = data.get("hourly")
        if not hourly:
            logger.warning("No hourly data in response")
            return pd.DataFrame()

        if "time" not in hourly:
            logger.warning("No 'time' field in hourly data")
            return pd.DataFrame()

        times = pd.to_datetime(hourly["time"], utc=True)

        frames = []
        for api_var, db_var in self.VARIABLE_MAPPING.items():
            values = hourly.get(api_var)
            if values is None:
                continue  # pragma: no cover  # Coverage tools may not mark continue in loop as covered, even when exercised by tests
            s = pd.Series(values, index=times, name="value").dropna()
            if s.empty:
                continue
            dfv = s.to_frame()
            dfv["timestamp"] = dfv.index
            dfv["variable"] = db_var
            if is_forecast and issue_time is not None:
                dfv["issue_time"] = issue_time
            frames.append(dfv.reset_index(drop=True))

        if not frames:
            logger.warning("No valid records parsed from API response")
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)
