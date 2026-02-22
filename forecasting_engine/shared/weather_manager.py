from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Tuple

import pandas as pd

from forecasting_db.models import WeatherVariable
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.weather_actual_io import WeatherActualIO
from forecasting_engine.db_io.weather_forecast_io import WeatherForecastIO
from forecasting_engine.db_io.weather_site_io import WeatherSitesIO
from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.shared.openmeteo_client import OpenMeteoClient

logger = get_logger(__name__)


class WeatherManager:
    """
    Manages weather data:
      1) Assign weather_site_ids to assets (grid-based)
      2) Fetch + store forecasts and historical actuals for each weather site
    """

    WEATHER_GRID_RESOLUTION = 0.05  # degrees

    def __init__(
        self,
        assets_io: AssetsIO,
        weather_sites_io: WeatherSitesIO,
        weather_actual_io: WeatherActualIO | None = None,
        weather_forecast_io: WeatherForecastIO | None = None,
        client: OpenMeteoClient | None = None,
    ):
        self.assets_io = assets_io
        self.weather_sites_io = weather_sites_io
        self.weather_actual_io = weather_actual_io
        self.weather_forecast_io = weather_forecast_io
        self.client = client or OpenMeteoClient()

    @staticmethod
    def _derive_weather_site(
        lat: float, lon: float, res: float = 0.05
    ) -> Tuple[str, float, float]:
        """
        Deterministically map (lat, lon) to a grid cell ID and query coordinates.

        Returns:
            weather_site_id: str   # stable identifier for grid cell
            site_lat: float        # center latitude of grid cell
            site_long: float       # center longitude of grid cell
        """
        if lat is None or lon is None:
            raise ValueError("Latitude and longitude must not be None.")

        # Use integer indices to avoid float string drift
        lat_i = math.floor(lat / res)
        lon_i = math.floor(lon / res)

        cell_lat = lat_i * res
        cell_lon = lon_i * res

        site_lat = round(cell_lat + res / 2, 6)
        site_long = round(cell_lon + res / 2, 6)

        weather_site_id = f"cell_{lat_i}_{lon_i}"
        return weather_site_id, site_lat, site_long

    def assign_db_assets_weather_sites(self) -> Dict[str, int]:
        assets = self.assets_io.list_assets_with_coords()
        total = len(assets)

        stats = {
            "total": total,
            "assigned": 0,
            "skipped": 0,
            "failed": 0,
            "site_upserted": 0,
        }
        logger.info("Assigning weather_site_id for %s assets", total)

        for asset in assets:
            try:
                if getattr(asset, "weather_site_id", None):
                    stats["skipped"] += 1
                    continue

                weather_site_id, site_lat, site_long = self._derive_weather_site(
                    asset.latitude,
                    asset.longitude,
                    self.WEATHER_GRID_RESOLUTION,
                )

                upserted = self.weather_sites_io.upsert_site(
                    weather_site_id, site_lat, site_long
                )
                if upserted:
                    stats["site_upserted"] += 1

                self.assets_io.update_weather_site_id(asset.asset_uuid, weather_site_id)
                stats["assigned"] += 1

            except Exception:
                stats["failed"] += 1
                logger.exception(
                    "Failed to assign weather_site_id for asset=%s (lat=%s lon=%s)",
                    getattr(asset, "asset_uuid", None),
                    getattr(asset, "latitude", None),
                    getattr(asset, "longitude", None),
                )

        logger.info("Weather site assignment done. Stats=%s", stats)
        return stats

    def fetch_and_store_weather_data(self, forecast_hours: int = 60) -> Dict[str, int]:
        """
        Fetch and store weather forecasts and historical data for all weather sites.

        Daily behavior:
          - forecast: next `forecast_hours` hours for each site
          - historical: backfill any missing actuals since latest timestamp per variable
            (safe to re-run due to ON CONFLICT DO NOTHING in IO layer)
        """
        if not self.weather_actual_io or not self.weather_forecast_io:
            logger.error(
                "WeatherActualIO and WeatherForecastIO not initialized. Cannot fetch weather data."
            )
            return {
                "total_sites": 0,
                "forecast_records": 0,
                "historical_records": 0,
                "failed": 0,
            }

        sites = self.weather_sites_io.list_sites()
        total = len(sites)

        stats = {
            "total_sites": total,
            "forecast_records": 0,
            "historical_records": 0,
            "failed": 0,
        }
        logger.info("Fetching weather data for %s sites", total)

        for site in sites:
            try:
                stats["forecast_records"] += self._fetch_and_store_forecast_for_site(
                    site, forecast_hours
                )
                stats[
                    "historical_records"
                ] += self._fetch_and_store_historical_for_site(site)

            except Exception:
                stats["failed"] += 1
                logger.exception(
                    "Failed to fetch weather data for site=%s (lat=%s lon=%s)",
                    getattr(site, "weather_site_id", None),
                    getattr(site, "site_lat", None),
                    getattr(site, "site_long", None),
                )

        logger.info("Weather data fetch completed. Stats=%s", stats)
        return stats

    def _fetch_and_store_forecast_for_site(self, site, forecast_hours: int) -> int:
        forecast_df = self.client.fetch_forecast(
            latitude=site.site_lat,
            longitude=site.site_long,
            hours=forecast_hours,
        )
        if forecast_df.empty:
            return 0

        forecast_df = forecast_df.copy()
        forecast_df["weather_site_id"] = site.weather_site_id

        inserted = self.weather_forecast_io.bulk_insert_forecasts(
            forecast_df.to_dict(orient="records")
        )
        logger.debug(
            "Inserted %s forecast records for %s", inserted, site.weather_site_id
        )
        return inserted

    def _fetch_and_store_historical_for_site(self, site) -> int:
        """
        Fetch historical data once per site, then filter per variable against latest timestamps.
        """
        latest_by_var = self.weather_actual_io.get_latest_reading_timestamps(
            site.weather_site_id
        )
        latest_values = [ts for ts in latest_by_var.values() if ts is not None]
        earliest_latest = min(latest_values) if latest_values else None

        historical_df = self.client.fetch_historical(
            latitude=site.site_lat,
            longitude=site.site_long,
            end_date=datetime.now(timezone.utc),
            start_date=earliest_latest,
        )
        if historical_df.empty:
            return 0

        # Attach site id
        historical_df = historical_df.copy()
        historical_df["weather_site_id"] = site.weather_site_id

        keep_frames: list[pd.DataFrame] = []

        for var in WeatherVariable:
            dfv = historical_df[historical_df["variable"] == var.value]
            if dfv.empty:
                continue

            latest_ts = latest_by_var.get(var)
            if latest_ts is not None:
                dfv = dfv[dfv["timestamp"] > latest_ts]
                if dfv.empty:
                    continue

            dfv = dfv.copy()
            dfv["variable"] = var  # store as enum (matches DB column type)
            keep_frames.append(
                dfv[["weather_site_id", "timestamp", "variable", "value"]]
            )

        if not keep_frames:
            return 0

        records = pd.concat(keep_frames, ignore_index=True).to_dict(orient="records")
        inserted = self.weather_actual_io.bulk_insert_readings(records)

        logger.debug(
            "Inserted %s historical records for %s", inserted, site.weather_site_id
        )
        return inserted
