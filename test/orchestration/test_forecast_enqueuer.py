# test_forecast_enqueuer.py
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone

from forecasting_engine.orchestration import forecast_enqueuer as enqueuer


@pytest.fixture
def fake_measurement_io():
    return MagicMock()


@pytest.fixture
def fake_forecast_run_io():
    return MagicMock()


@pytest.fixture
def fake_sqs_client():
    with patch(
        "forecasting_engine.orchestration.forecast_enqueuer.SQS_CLIENT"
    ) as mock_sqs:
        yield mock_sqs


@pytest.fixture
def asset_times():
    now = datetime.now(timezone.utc)
    return {
        "asset1": (now - timedelta(hours=1), 100),
        "asset2": (now - timedelta(hours=2), 200),
        "asset3": (now - timedelta(hours=3), 300),
    }


# ----------------------------
# Tests for get_assets_to_forecast
# ----------------------------
def test_get_assets_to_forecast_includes_newer_measurement(
    fake_measurement_io, fake_forecast_run_io, asset_times
):
    fake_measurement_io.get_latest_load_per_asset.return_value = asset_times
    # last forecast times: only asset1 is up to date
    fake_forecast_run_io.get_latest_forecast_run_per_asset.return_value = {
        "asset1": asset_times["asset1"][0],
        "asset2": asset_times["asset2"][0]
        - timedelta(minutes=5),  # older than measurement
        # asset3 has no forecast
    }

    result = enqueuer.get_assets_to_forecast(fake_measurement_io, fake_forecast_run_io)

    assert "asset1" not in result
    assert "asset2" in result
    assert "asset3" in result


def test_get_assets_to_forecast_no_measurements(
    fake_measurement_io, fake_forecast_run_io
):
    fake_measurement_io.get_latest_load_per_asset.return_value = {}
    result = enqueuer.get_assets_to_forecast(fake_measurement_io, fake_forecast_run_io)
    assert result == {}


# ----------------------------
# Tests for enqueue_assets
# ----------------------------
def test_enqueue_assets_sends_sqs_messages(fake_sqs_client):
    assets_to_forecast = {
        "assetA": datetime(2025, 10, 26, 12, 0),
        "assetB": datetime(2025, 10, 26, 13, 0),
    }

    enqueuer.enqueue_assets(assets_to_forecast)

    calls = [
        (
            (),
            {
                "QueueUrl": enqueuer.QUEUE_URL,
                "MessageBody": json.dumps({"asset_id": "assetA"}),
            },
        ),
        (
            (),
            {
                "QueueUrl": enqueuer.QUEUE_URL,
                "MessageBody": json.dumps({"asset_id": "assetB"}),
            },
        ),
    ]

    fake_sqs_client.send_message.assert_has_calls(calls, any_order=True)
    assert fake_sqs_client.send_message.call_count == 2


# ----------------------------
# Tests for enqueue_new_forecasts
# ----------------------------
def test_enqueue_new_forecasts_calls_enqueue_assets(monkeypatch, asset_times):
    # Mock SessionLocal context manager
    mock_session = MagicMock()
    monkeypatch.setattr(enqueuer, "SessionLocal", lambda: mock_session)

    fake_measurement_io = MagicMock()
    fake_forecast_run_io = MagicMock()
    fake_measurement_io.get_latest_load_per_asset.return_value = asset_times
    fake_forecast_run_io.get_latest_forecast_run_per_asset.return_value = {}

    # Patch MeasurementsIO and ForecastRunIO constructors to return our fakes
    monkeypatch.setattr(enqueuer, "MeasurementsIO", lambda session: fake_measurement_io)
    monkeypatch.setattr(enqueuer, "ForecastRunIO", lambda session: fake_forecast_run_io)

    fake_assets_to_forecast = {"asset1": asset_times["asset1"][0]}
    monkeypatch.setattr(
        enqueuer, "get_assets_to_forecast", lambda m, f: fake_assets_to_forecast
    )

    # Patch enqueue_assets to track calls
    mock_enqueue_assets = MagicMock()
    monkeypatch.setattr(enqueuer, "enqueue_assets", mock_enqueue_assets)

    enqueuer.enqueue_new_forecasts()

    mock_enqueue_assets.assert_called_once_with(fake_assets_to_forecast)


def test_enqueue_new_forecasts_no_assets(monkeypatch):
    mock_session = MagicMock()
    monkeypatch.setattr(enqueuer, "SessionLocal", lambda: mock_session)
    fake_measurement_io = MagicMock()
    fake_forecast_run_io = MagicMock()
    monkeypatch.setattr(enqueuer, "MeasurementsIO", lambda session: fake_measurement_io)
    monkeypatch.setattr(enqueuer, "ForecastRunIO", lambda session: fake_forecast_run_io)
    # No assets to forecast
    monkeypatch.setattr(enqueuer, "get_assets_to_forecast", lambda m, f: {})

    with patch(
        "forecasting_engine.orchestration.forecast_enqueuer.logger"
    ) as mock_logger:
        enqueuer.enqueue_new_forecasts()
        mock_logger.warning.assert_called_with("No assets require a forecast")
