import json
import pytest
from unittest.mock import patch

import forecasting_engine.services.forecast_poller.consumer as forecast_consumer


@pytest.fixture
def fake_message():
    return {
        "Body": json.dumps({"asset_id": "12345"}),
        "ReceiptHandle": "fake-receipt-handle",
    }


@pytest.fixture
def mock_logger():
    with patch.object(forecast_consumer, "logger") as mock_log:
        yield mock_log


@pytest.fixture
def mock_sqs_delete():
    with patch.object(forecast_consumer.SQS_CLIENT, "delete_message") as mock:
        yield mock


@pytest.fixture
def mock_run_forecast():
    with patch.object(forecast_consumer, "run_asset_forecast") as mock:
        yield mock


class TestProcessMeasurementQueueSuccess:
    def test_successful_processing(
        self, fake_message, mock_logger, mock_sqs_delete, mock_run_forecast
    ):
        mock_run_forecast.return_value = {"status": "success"}

        result = forecast_consumer.process_measurement_queue_message(fake_message)

        assert result is True
        mock_run_forecast.assert_called_once_with("12345")
        mock_sqs_delete.assert_called_once_with(
            QueueUrl=forecast_consumer.QUEUE_URL, ReceiptHandle="fake-receipt-handle"
        )
        mock_logger.info.assert_any_call("✅ Processed and deleted message for 12345")


class TestProcessMeasurementQueueFailure:
    def test_forecast_failure(
        self, fake_message, mock_logger, mock_sqs_delete, mock_run_forecast
    ):
        mock_run_forecast.return_value = {"status": "failed"}

        result = forecast_consumer.process_measurement_queue_message(fake_message)

        assert result is False
        mock_sqs_delete.assert_not_called()
        mock_logger.error.assert_any_call("⚠️ Forecast failed for 12345, status: failed")


class TestProcessMeasurementQueueException:
    def test_exception_in_processing(
        self, fake_message, mock_logger, mock_sqs_delete, mock_run_forecast
    ):
        mock_run_forecast.side_effect = ValueError("Something went wrong")

        result = forecast_consumer.process_measurement_queue_message(fake_message)

        assert result is False
        mock_logger.error.assert_any_call(
            "Error processing message: Something went wrong"
        )
        mock_sqs_delete.assert_not_called()
