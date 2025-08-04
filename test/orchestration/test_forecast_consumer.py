# test_forecast_consumer.py
import json
import pytest
from unittest.mock import patch

import forecasting_engine.orchestration.forecast_consumer as forecast_consumer


@pytest.fixture
def fake_message():
    return {
        "Body": json.dumps(
            {"Records": [{"s3": {"object": {"key": "assets/12345/data.csv"}}}]}
        ),
        "ReceiptHandle": "fake-receipt-handle",
    }


@pytest.fixture
def mock_logger():
    with patch.object(forecast_consumer, "logger") as mock_log:
        yield mock_log


@pytest.fixture
def mock_extract_asset_id():
    with patch.object(
        forecast_consumer, "extract_asset_id_from_s3_key", return_value="12345"
    ) as mock:
        yield mock


@pytest.fixture
def mock_sqs_delete():
    with patch.object(forecast_consumer.SQS_CLIENT, "delete_message") as mock:
        yield mock


@pytest.fixture
def mock_run_forecast():
    with patch.object(forecast_consumer, "run_asset_forecast") as mock:
        yield mock


class TestProcessTrainingQueueSuccess:
    def test_successful_processing(
        self,
        fake_message,
        mock_logger,
        mock_extract_asset_id,
        mock_sqs_delete,
        mock_run_forecast,
    ):
        mock_run_forecast.return_value = {"status": "success"}

        result = forecast_consumer.process_training_queue_message(fake_message)

        assert result is True
        mock_extract_asset_id.assert_called_once_with("assets/12345/data.csv")
        mock_run_forecast.assert_called_once_with("12345")
        mock_sqs_delete.assert_called_once_with(
            QueueUrl=forecast_consumer.QUEUE_URL, ReceiptHandle="fake-receipt-handle"
        )
        mock_logger.info.assert_any_call("✅ Processed and deleted message for 12345")


class TestProcessTrainingQueueFailure:
    def test_forecast_failure(
        self,
        fake_message,
        mock_logger,
        mock_extract_asset_id,
        mock_sqs_delete,
        mock_run_forecast,
    ):
        mock_run_forecast.return_value = {"status": "failed"}

        result = forecast_consumer.process_training_queue_message(fake_message)

        assert result is False
        mock_sqs_delete.assert_not_called()
        mock_logger.error.assert_any_call("⚠️ Forecast failed for 12345, status: failed")


class TestProcessTrainingQueueException:
    def test_exception_in_processing(
        self,
        fake_message,
        mock_logger,
        mock_extract_asset_id,
        mock_sqs_delete,
        mock_run_forecast,
    ):
        mock_extract_asset_id.side_effect = ValueError("Bad S3 key")

        result = forecast_consumer.process_training_queue_message(fake_message)

        assert result is False
        mock_logger.error.assert_any_call("Error processing message: Bad S3 key")
        mock_sqs_delete.assert_not_called()
