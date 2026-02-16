from forecasting_engine.shared.weather_manager import WeatherManager
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.weather_site_io import WeatherSitesIO
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.shared.logger_factory import get_logger

logger = get_logger(__name__)


def main():
    """
    Main entry point for weather service.
    """

    session = SessionLocal()

    try:
        assets_io = AssetsIO(session)
        weather_sites_io = WeatherSitesIO(session)
        manager = WeatherManager(assets_io, weather_sites_io)
        manager.assign_db_assets_weather_sites()
    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    main()
