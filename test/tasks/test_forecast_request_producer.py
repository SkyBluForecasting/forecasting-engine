import pytest
import logging
from unittest.mock import patch
from forecasting_engine.tasks import forecast_request_producer


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    caplog.set_level(logging.INFO)


@patch("forecasting_engine.tasks.forecast_request_producer.enqueue_new_forecasts")
@patch(
    "forecasting_engine.tasks.forecast_request_producer.time.sleep", return_value=None
)
def test_main_run_once(mock_sleep, mock_enqueue, caplog):
    """Test one iteration of main() when run_once=True."""
    forecast_request_producer.main(run_once=True, interval=123)

    mock_enqueue.assert_called_once()
    mock_sleep.assert_not_called()
    assert "Finished enqueuing messages for forecast generation." in caplog.text


@patch(
    "forecasting_engine.tasks.forecast_request_producer.enqueue_new_forecasts",
    side_effect=Exception("boom"),
)
@patch(
    "forecasting_engine.tasks.forecast_request_producer.time.sleep", return_value=None
)
def test_main_logs_error_and_continues(mock_sleep, mock_enqueue, caplog):
    """Test error handling still logs and continues looping."""
    # Run only one iteration (avoid infinite loop)
    forecast_request_producer.main(run_once=True, interval=60)

    mock_enqueue.assert_called_once()
    assert "Error enqueuing messages for forecast generation: boom" in caplog.text


@patch("forecasting_engine.tasks.forecast_request_producer.enqueue_new_forecasts")
@patch(
    "forecasting_engine.tasks.forecast_request_producer.time.sleep",
    side_effect=KeyboardInterrupt,
)
def test_main_continuous_loop_sleeps(mock_sleep, mock_enqueue, caplog):
    """Test that when run_once=False, it sleeps after each iteration."""
    with pytest.raises(KeyboardInterrupt):
        forecast_request_producer.main(run_once=False, interval=42)

    mock_enqueue.assert_called_once()
    mock_sleep.assert_called_with(42)
    assert "Sleeping 42 seconds before next run..." in caplog.text
