import time
from forecasting_engine.orchestration.forecast_consumer import (
    process_training_queue_message,
)
from forecasting_engine.config import SQS_CLIENT, QUEUE_URL


def process_once():
    """Process one SQS receive_message cycle."""
    response = SQS_CLIENT.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=20,
    )

    messages = response.get("Messages", [])
    if not messages:
        return False

    for msg in messages:
        process_training_queue_message(msg)

    return True


def main():
    while True:
        process_once()
        time.sleep(1)  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    main()
