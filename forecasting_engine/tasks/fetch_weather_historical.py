"""
Entrypoint for fetching historical weather data from Open-Meteo.

This script can be called directly via CLI to get historical weather data for a specific location
without requiring database access.

Example:
    python -m forecasting_engine.tasks.fetch_weather_historical 52.5 13.4
    python -m forecasting_engine.tasks.fetch_weather_historical 52.5 13.4 --start-date 2026-02-01
    python -m forecasting_engine.tasks.fetch_weather_historical 52.5 13.4 --start-date 2026-02-01 --format csv
"""

import argparse
import sys
from datetime import datetime, timezone
from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.shared.openmeteo_client import OpenMeteoClient

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical weather data for a specific location."
    )
    parser.add_argument("latitude", type=float, help="Latitude of the location.")
    parser.add_argument("longitude", type=float, help="Longitude of the location.")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for historical data (format: YYYY-MM-DD). Default: last 30 days.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End datetime for historical data (format: YYYY-MM-DD HH:MM:SS). Default: now.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "text"],
        default="text",
        help="Output format (default: text).",
    )

    args = parser.parse_args()

    try:
        # Parse start_date if provided
        start_date = None
        if args.start_date:
            try:
                start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid date format: {args.start_date}. Use YYYY-MM-DD.")
                return 2

        # Parse end_date if provided, otherwise use now
        end_date = datetime.now(timezone.utc)
        if args.end_date:
            try:
                end_date = datetime.strptime(
                    args.end_date, "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                logger.error(
                    f"Invalid datetime format: {args.end_date}. Use YYYY-MM-DD HH:MM:SS."
                )
                return 2

        client = OpenMeteoClient()
        logger.info(
            f"Fetching historical data for lat={args.latitude}, lon={args.longitude}"
        )
        if start_date:
            logger.info(f"Starting from {start_date.date()}")
        logger.info(f"Ending at {end_date}")

        historical_df = client.fetch_historical(
            latitude=args.latitude,
            longitude=args.longitude,
            end_date=end_date,
            start_date=start_date,
        )

        if historical_df.empty:
            logger.warning("No historical data returned.")
            return 1

        # Output in requested format
        if args.format == "csv":
            print(historical_df.to_csv(index=False))
        elif args.format == "json":
            print(historical_df.to_json(orient="records", date_format="iso"))
        else:  # text
            print(historical_df.to_string(index=False))

        logger.info(f"Successfully retrieved {len(historical_df)} historical records.")
        return 0

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Error fetching historical data: {e}")
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
