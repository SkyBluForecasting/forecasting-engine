import json
from forecasting_engine.tasks.run_single_forecast import run_asset_forecast
from forecasting_engine.orchestration.s3_utils import extract_asset_id_from_s3_key
from forecasting_engine.config import SQS_CLIENT, QUEUE_URL
from forecasting_engine.orchestration.logger_factory import get_logger

logger = get_logger(__name__)


def extract_s3_key_from_message(message: dict) -> str:
    body = json.loads(message["Body"])
    return body["Records"][0]["s3"]["object"]["key"]


def delete_sqs_message(receipt_handle: str):
    SQS_CLIENT.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)


def process_training_queue_message(message: dict) -> bool:
    try:
        s3_key = extract_s3_key_from_message(message)
        asset_id = extract_asset_id_from_s3_key(s3_key)
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
