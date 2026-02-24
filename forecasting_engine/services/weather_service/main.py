from forecasting_engine.shared.weather_manager import WeatherManager
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.weather_site_io import WeatherSitesIO
from forecasting_engine.db_io.weather_actual_io import WeatherActualIO
from forecasting_engine.db_io.weather_forecast_io import WeatherForecastIO
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.shared.logger_factory import get_logger

logger = get_logger(__name__)


def main():
    """
    Main entry point for weather service.

    Performs two tasks:
    1. Assigns weather_site_id to assets that don't have one
    2. Fetches and stores weather forecasts and historical data for all weather sites
    """

    session = SessionLocal()

    try:
        assets_io = AssetsIO(session)
        weather_sites_io = WeatherSitesIO(session)
        weather_actual_io = WeatherActualIO(session)
        weather_forecast_io = WeatherForecastIO(session)

        manager = WeatherManager(
            assets_io,
            weather_sites_io,
            weather_actual_io,
            weather_forecast_io,
        )

        logger.info("Starting weather site assignment")
        manager.assign_db_assets_weather_sites()

        logger.info("Starting weather data fetch")
        manager.fetch_and_store_weather_data()

    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    main()
