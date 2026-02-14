import argparse
from forecasting_engine.shared.weather_site_manager import WeatherSiteManager
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.shared.logger_factory import get_logger

logger = get_logger(__name__)
    


def main(
    manager_cls=WeatherSiteManager,
    assets_io_cls=AssetsIO,
    session_cls=SessionLocal,
    logger_factory=get_logger,
):  # pragma: no cover
    """
    Main entry point for weather service.
    """

    session = session_cls()
    assets_io = assets_io_cls(session)
    manager = manager_cls(assets_io)

    try:
        manager.assign_db_assets_weather_sites()
    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    main()
