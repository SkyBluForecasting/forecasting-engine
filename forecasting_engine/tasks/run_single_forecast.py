"""
Entrypoint for generating a forecast for a single asset.

This script can be:
- Called directly via CLI with an asset ID as an argument.
- Imported and used via the `run_asset_forecast()` function in a polling or batch-processing script.

Designed to be safe for orchestration use (e.g., EventBridge, polling scripts).
"""

import argparse
import sys
from forecasting_engine.shared.forecast_runner import ForecastManager
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.shared.logger_factory import get_logger

logger = get_logger(__name__)


def run_asset_forecast(asset_id: str) -> dict:
    """
    Run forecast generation for a specific asset.

    Args:
        asset_id (str): The unique identifier of the asset to forecast.

    Returns:
        dict: A structured result dictionary with the following keys:
            - 'asset_id' (str): The ID of the asset processed.
            - 'status' (str): One of:
                * "success"      — Forecast was generated successfully.
                * "not_found"    — Missing required input (FileNotFoundError).
                * "bad_input"    — Invalid or missing measurements (ValueError).
                * "fatal_error"  — Unexpected exception.
            - 'message' (str or None): Human-readable error message if applicable.
    """
    try:
        with SessionLocal() as session:
            fm = ForecastManager(session)
            fm.generate_forecast(asset_id)
        return {
            "asset_id": asset_id,
            "status": "success",
            "message": None,
        }

    except FileNotFoundError as e:
        logger.error(f"[NOT FOUND] {e}")
        return {"asset_id": asset_id, "status": "not_found", "message": str(e)}

    except ValueError as e:
        logger.error(f"[BAD INPUT] {e}")
        return {"asset_id": asset_id, "status": "bad_input", "message": str(e)}

    except Exception as e:
        logger.exception(f"[FATAL] Unexpected error for asset {asset_id}: {e}")
        return {"asset_id": asset_id, "status": "fatal_error", "message": str(e)}


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
