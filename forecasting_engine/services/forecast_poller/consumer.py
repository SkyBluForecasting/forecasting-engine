import json
from forecasting_engine.shared.forecast_runner import run_asset_forecast
from forecasting_engine.config import SQS_CLIENT, QUEUE_URL
from forecasting_engine.shared.logger_factory import get_logger

logger = get_logger(__name__)


def delete_sqs_message(receipt_handle: str):
    SQS_CLIENT.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)


def process_measurement_queue_message(message: dict) -> bool:
    try:
        body = json.loads(message["Body"])
        asset_id = body["asset_id"]
        logger.info(f"📦 Received message for asset: {asset_id}")

        result = run_asset_forecast(asset_id)

        if result["status"] == "success":
            delete_sqs_message(message["ReceiptHandle"])
            logger.info(f"✅ Processed and deleted message for {asset_id}")
            return True
        else:
            logger.error(
                f"⚠️ Forecast failed for {asset_id}, status: {result['status']}"
            )
            return False
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return False
