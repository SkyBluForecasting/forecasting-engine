from typing import List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from forecasting_db.models import WeatherSite
from .base_io import BaseIO


class WeatherSitesIO(BaseIO):
    """Handles loading and writing weather site metadata."""

    def __init__(self, session: Session):
        super().__init__(session)

    def list_sites(self) -> List[WeatherSite]:
        """Return all weather sites."""
        return self.session.query(WeatherSite).all()

    def get_site(self, weather_site_id: str) -> Optional[WeatherSite]:
        """Fetch a single weather site by ID."""
        return (
            self.session.query(WeatherSite)
            .filter(WeatherSite.weather_site_id == weather_site_id)
            .one_or_none()
        )

    def exists(self, weather_site_id: str) -> bool:
        """Return True if the weather site exists."""
        return self.get_site(weather_site_id) is not None

    def upsert_site(
        self, weather_site_id: str, site_lat: float, site_long: float
    ) -> bool:
        """
        Idempotently insert a weather site.

        Returns:
            True if inserted, False if it already existed.
        """
        existing = self.get_site(weather_site_id)
        if existing:
            return False

        site = WeatherSite(
            weather_site_id=weather_site_id,
            site_lat=site_lat,
            site_long=site_long,
        )
        self.session.add(site)
        self.session.commit()
        return True

    def from_df(self, df: pd.DataFrame, *args, **kwargs):
        """Write weather sites from DataFrame to DB (not used yet)."""
        raise NotImplementedError("from_df() not implemented for WeatherSitesIO")

    def to_df(self) -> pd.DataFrame:
        """Load weather sites into a pandas DataFrame."""
        sites = self.list_sites()
        if not sites:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "weather_site_id": s.weather_site_id,
                    "site_lat": s.site_lat,
                    "site_long": s.site_long,
                }
                for s in sites
            ]
        )
