"""	Run forecast for a single asset (call orchestration function with asset_id from CLI arg)."""

import argparse
import logging
import sys
from forecasting_engine.orchestration.forecast_runner import generate_forecast_for_asset
from forecasting_engine.orchestration.logger_factory import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run forecast for a specific asset.")
    parser.add_argument(
        "asset_id", type=str, help="Asset ID for which to generate the forecast."
    )
    args = parser.parse_args()

    try:
        generate_forecast_for_asset(args.asset_id)
    except FileNotFoundError as e:
        logger.error(f"[NOT FOUND] {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"[BAD INPUT] {e}")
        sys.exit(2)
    except Exception as e:
        logger.exception(f"[FATAL] Unexpected error for asset {args.asset_id}: {e}")
        sys.exit(99)


if __name__ == "__main__":
    main()
