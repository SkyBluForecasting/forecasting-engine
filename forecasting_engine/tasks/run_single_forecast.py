"""
Entrypoint for generating a forecast for a single asset.

This script can be:
- Called directly via CLI with an asset ID as an argument.
- Imported and used via the `run_asset_forecast()` function in a polling or batch-processing script.

It wraps `generate_forecast_for_asset()` and returns structured results for robust error handling.
Designed to be safe for orchestration use (e.g., EventBridge, polling scripts).
"""

import argparse
import sys
from forecasting_engine.orchestration.forecast_runner import generate_forecast_for_asset
from forecasting_engine.orchestration.logger_factory import get_logger

logger = get_logger(__name__)


def run_asset_forecast(asset_id: str) -> dict:
    """
    Run forecast generation for a specific asset.

    Args:
        asset_id (str): The unique identifier of the asset to forecast.

    Returns:
        dict: A structured result dictionary with the following keys:
            - 'asset_id' (str): The ID of the asset processed.
            - 'status' (str): One of the following status values:
                * "success"      — Forecast was generated successfully.
                * "not_found"    — A required file or input was missing (FileNotFoundError).
                * "bad_input"    — Asset input was invalid or improperly formatted (ValueError).
                * "fatal_error"  — An unexpected error occurred (generic Exception).
            - 'message' (str or None): A human-readable error message if applicable, otherwise None.

    Side Effects:
        - Logs messages at ERROR or EXCEPTION level using the system logger.

    Notes:
        This function is designed to be safely called in a polling or batch-processing context.
        It should not raise uncaught exceptions or call sys.exit().
    """
    try:
        generate_forecast_for_asset(asset_id)
        return {
            "asset_id": asset_id,
            "status": "success",
            "message": None,
        }
    except FileNotFoundError as e:
        logger.error(f"[NOT FOUND] {e}")
        return {
            "asset_id": asset_id,
            "status": "not_found",
            "message": str(e),
        }
    except ValueError as e:
        logger.error(f"[BAD INPUT] {e}")
        return {
            "asset_id": asset_id,
            "status": "bad_input",
            "message": str(e),
        }
    except Exception as e:
        logger.exception(f"[FATAL] Unexpected error for asset {asset_id}: {e}")
        return {
            "asset_id": asset_id,
            "status": "fatal_error",
            "message": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run forecast for a specific asset.")
    parser.add_argument(
        "asset_id", type=str, help="Asset ID for which to generate the forecast."
    )
    args = parser.parse_args()
    exit_code = run_asset_forecast(args.asset_id)
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
