import argparse
import time
import logging
from forecasting_engine.orchestration.forecast_consumer import (
    process_measurement_queue_message,
)
from forecasting_engine.config import SQS_CLIENT, QUEUE_URL

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def poll_sqs():
    """
    Poll SQS queue once and process up to 10 messages.
    Returns the number of messages processed.
    """
    try:
        response = SQS_CLIENT.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,  # batch size per poll
            WaitTimeSeconds=20,  # long poll
            VisibilityTimeout=60,  # allow enough time to process
        )
        messages = response.get("Messages", [])

        if not messages:
            return 0

        for msg in messages:
            success = process_measurement_queue_message(msg)
            if not success:
                logger.warning(
                    "Message was not processed successfully, leaving in queue."
                )

        return len(messages)

    except Exception as e:
        logger.error(f"Error polling SQS: {e}")
        return 0


def main(run_once: bool = False):
    """
    Periodic poller loop with backoff.
    """
    while True:
        num_processed = poll_sqs()
        if num_processed == 0:
            logger.info("No messages found. Sleeping 5 minutes...")
            time.sleep(300)  # 5 minutes if queue is empty
        else:
            logger.info(
                f"Processed {num_processed} messages. Checking again shortly..."
            )
            time.sleep(10)  # short pause to avoid tight loop


if __name__ == "__main__":  # pragma: no coverage
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once", action="store_true", help="Run one iteration and exit"
    )
    args = parser.parse_args()
    main(run_once=args.once)
