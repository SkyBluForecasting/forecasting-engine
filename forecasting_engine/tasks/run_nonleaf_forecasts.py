"""
Entrypoint for running forecasts for all non-leaf assets bottom-up.

Called daily by cron. Ensures all assets at depth D are run before any at depth D-1.

Behavior:
- Always continues after per-asset failures (best-effort).
- Exits with code 0 only if all assets succeeded; otherwise exits 3.
"""

import argparse
import sys

from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.shared.forecast_runner import run_asset_forecast
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.db_io.assets_io import AssetsIO

logger = get_logger(__name__)


def run_bottom_up(*, dry_run: bool = False, asset_type: str | None = None) -> dict:
    """Run all non-leaf forecasts bottom-up, continuing after failures."""
    session = SessionLocal()
    try:
        assets_io = AssetsIO(session)

        max_depth = assets_io.get_max_depth_non_leaf(asset_type=asset_type)
        logger.info(f"Max non-leaf depth: {max_depth}")

        total = 0
        failures: list[dict] = []
        skipped: list[dict] = []

        for depth in range(max_depth, -1, -1):
            parents = assets_io.list_non_leaf_assets_at_depth(
                depth, asset_type=asset_type
            )
            if not parents:
                continue

            logger.info(
                f"Running non-leaf forecasts at depth={depth} count={len(parents)}"
            )

            for asset in parents:
                total += 1

                if dry_run:
                    logger.info(
                        f"[dry-run] Would run asset={asset.asset_uuid} depth={depth}"
                    )
                    continue

                result = run_asset_forecast(asset.asset_uuid)
                status = result.get("status")

                if status == "skipped":
                    skipped.append(
                        {
                            "asset_uuid": asset.asset_uuid,
                            "depth": depth,
                            "message": result.get("message"),
                        }
                    )
                    logger.warning(
                        f"Forecast skipped: asset={asset.asset_uuid} depth={depth} msg={result.get('message')}"
                    )
                    continue

                if status != "success":
                    failure = {
                        "asset_uuid": asset.asset_uuid,
                        "depth": depth,
                        "status": status,
                    }
                    failures.append(failure)
                    logger.error(f"Forecast failed: {failure}")

        overall_status = "success" if not failures else "fatal_error"
        return {
            "status": overall_status,
            "total_attempted": total,
            "failures": failures,
            "skipped": skipped,
        }
    except Exception:
        logger.exception("Bottom-up non-leaf forecast run failed")
        return {"status": "fatal_error"}
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Run non-leaf forecasts bottom-up.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Log what would run without running."
    )
    parser.add_argument(
        "--asset-type", type=str, default=None, help="Optional asset_type filter."
    )
    args = parser.parse_args()

    result = run_bottom_up(
        dry_run=args.dry_run,
        asset_type=args.asset_type,
    )
    logger.info(f"Run result: {result}")

    # Exit non-zero if any failures (so cron alerts you)
    status_to_exit_code = {"success": 0, "fatal_error": 3}
    sys.exit(status_to_exit_code.get(result["status"], 3))


if __name__ == "__main__":  # pragma: no cover
    main()
