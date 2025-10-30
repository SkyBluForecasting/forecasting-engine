import argparse
import time
import logging
from forecasting_engine.services.forecast_enqueuer.forecast_enqueuer import (
    enqueue_new_forecasts,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main(run_once: bool = False, interval: int = 1800):
    """
    Periodic loop which checks for new measurements in the DB, and then pushes a messsage to the
    measurements SQS queue for each applicable asset.
    """
    while True:
        try:
            enqueue_new_forecasts()
            logger.info("Finished enqueuing messages for forecast generation.")
        except Exception as e:
            logger.error(f"Error enqueuing messages for forecast generation: {e}")

        if run_once:
            break

        # Sleep before next iteration
        logger.info(f"Sleeping {interval} seconds before next run...")
        time.sleep(interval)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once", action="store_true", help="Run one iteration and exit"
    )
    parser.add_argument(
        "--interval", type=int, default=1800, help="Seconds between runs"
    )
    args = parser.parse_args()

    main(run_once=args.once, interval=args.interval)
