from typing import Dict
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.weather_site_io import WeatherSitesIO
from forecasting_engine.shared.logger_factory import get_logger
import math
from typing import Tuple

logger = get_logger(__name__)


class WeatherManager:
    """
    Manages weather data - assigning weather site ids, and fetching weather data.
    """

    RES_DEFAULT = 0.05 # hardcoded for now

    def __init__(self, assets_io: AssetsIO, weather_sites_io: WeatherSitesIO):
        """
        Initialize the WeatherManager.

        Args:
            assets_io: AssetsIO instance for database operations
        """
        self.assets_io = assets_io
        self.weather_sites_io = weather_sites_io


    @staticmethod
    def _derive_weather_site(
        lat: float,
        lon: float,
        res: float = 0.05
    ) -> Tuple[str, float, float]:
        """
        Derive a deterministic weather_site_id and its query coordinates.

        Returns:
            weather_site_id: str
            site_lat: float   # center latitude of grid cell
            site_lon: float   # center longitude of grid cell
        """

        if lat is None or lon is None:
            raise ValueError("Latitude and longitude must not be None.")

        # Lower-left grid corner
        cell_lat = math.floor(lat / res) * res
        cell_lon = math.floor(lon / res) * res

        # Query coordinate = center of cell
        site_lat = round(cell_lat + res / 2, 6)
        site_lon = round(cell_lon + res / 2, 6)

        weather_site_id = f"cell_{cell_lat:.2f}_{cell_lon:.2f}"

        return weather_site_id, site_lat, site_lon


    def assign_db_assets_weather_sites(self) -> Dict[str, int]:

        assets = self.assets_io.list_assets_with_coords()

        stats = {"assigned": 0, "skipped": 0, "failed": 0, "site_upserted": 0}

        for asset in assets:
            try:
                if asset.weather_site_id:
                    stats["skipped"] += 1
                    continue

                weather_site_id, site_lat, site_lon = self._derive_weather_site(
                    asset.latitude,
                    asset.longitude,
                    self.WEATHER_GRID_RESOLUTION,
                )

                upserted = self.weather_sites_io.upsert_site(
                    weather_site_id, site_lat, site_lon
                )

                if upserted:
                    stats["site_upserted"] += 1

                self.assets_io.update_weather_site_id(
                    asset.asset_uuid, weather_site_id
                )

                stats["assigned"] += 1

            except Exception:
                stats["failed"] += 1

        return stats
