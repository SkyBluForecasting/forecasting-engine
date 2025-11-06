"""
Entrypoint for generating a forecast for a single asset.

This script can be called directly via CLI with an asset ID as an argument.

"""

import argparse
import sys
from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.shared.forecast_runner import run_asset_forecast

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run forecast for a specific asset.")
    parser.add_argument(
        "asset_id", type=str, help="Asset ID for which to generate the forecast."
    )
    args = parser.parse_args()

    result = run_asset_forecast(args.asset_id)
    logger.info(f"Forecast run result: {result}")

    # Exit code mapping
    status_to_exit_code = {
        "success": 0,
        "not_found": 1,
        "bad_input": 2,
        "fatal_error": 3,
    }
    sys.exit(status_to_exit_code.get(result["status"], 3))


if __name__ == "__main__":  # pragma: no cover
    main()
