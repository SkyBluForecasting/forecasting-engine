#!/usr/bin/env python3
"""
Example script showing how to use WeatherSiteManager with an LNM file.

This script can be run standalone to assign weather_site_id to PV assets
based on a CSV file mapping asset IDs to weather site IDs.

Example LNM file format (CSV):
    asset_id,weather_site_id
    ASSET001,WS001
    ASSET002,WS002
    ...

Usage:
    python scripts/assign_weather_sites_from_lnm.py --lnm-file path/to/mapping.csv
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import forecasting_engine modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecasting_engine.shared.weather_site_manager import WeatherSiteManager
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.shared.logger_factory import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Assign weather_site_id to PV assets from LNM file"
    )
    parser.add_argument(
        "--lnm-file",
        type=str,
        required=True,
        help="Path to LNM file (CSV) with asset_id and weather_site_id columns",
    )
    parser.add_argument(
        "--asset-id-column",
        type=str,
        default="asset_id",
        help="Column name in LNM file containing asset IDs (default: asset_id)",
    )
    parser.add_argument(
        "--weather-site-id-column",
        type=str,
        default="weather_site_id",
        help="Column name in LNM file containing weather site IDs (default: weather_site_id)",
    )
    args = parser.parse_args()

    session = SessionLocal()
    assets_io = AssetsIO(session)
    manager = WeatherSiteManager(assets_io)

    try:
        stats = manager.assign_weather_sites_from_lnm_file(
            args.lnm_file,
            asset_id_column=args.asset_id_column,
            weather_site_id_column=args.weather_site_id_column,
        )
        logger.info(f"Assignment complete: {stats}")
    except Exception as e:
        logger.exception(f"Failed to assign weather sites: {e}")
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
