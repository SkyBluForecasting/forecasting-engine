"""
Entrypoint for fetching weather forecasts from Open-Meteo.

This script can be called directly via CLI to get weather forecasts for a specific location
without requiring database access.

Example:
    python -m forecasting_engine.tasks.fetch_weather_forecast 52.5 13.4
    python -m forecasting_engine.tasks.fetch_weather_forecast 52.5 13.4 --hours 120
"""

import argparse
import sys
from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.shared.openmeteo_client import OpenMeteoClient

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch weather forecast for a specific location."
    )
    parser.add_argument("latitude", type=float, help="Latitude of the location.")
    parser.add_argument("longitude", type=float, help="Longitude of the location.")
    parser.add_argument(
        "--hours",
        type=int,
        default=60,
        help="Number of hours to forecast (default: 60).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "text"],
        default="text",
        help="Output format (default: text).",
    )

    args = parser.parse_args()

    try:
        client = OpenMeteoClient()
        logger.info(
            f"Fetching {args.hours}-hour forecast for lat={args.latitude}, lon={args.longitude}"
        )

        forecast_df = client.fetch_forecast(
            latitude=args.latitude, longitude=args.longitude, hours=args.hours
        )

        if forecast_df.empty:
            logger.warning("No forecast data returned.")
            return 1

        # Output in requested format
        if args.format == "csv":
            print(forecast_df.to_csv(index=False))
        elif args.format == "json":
            print(forecast_df.to_json(orient="records", date_format="iso"))
        else:  # text
            print(forecast_df.to_string(index=False))

        logger.info(f"Successfully retrieved {len(forecast_df)} forecast records.")
        return 0

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Error fetching forecast: {e}")
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
