import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone

from forecasting_engine.services.forecast_enqueuer import forecast_enqueuer as enqueuer


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def fake_measurement_io():
    return MagicMock()


@pytest.fixture
def fake_forecast_run_io():
    return MagicMock()


@pytest.fixture
def fake_assets_io():
    return MagicMock()


@pytest.fixture
def fake_sqs_client():
    with patch(
        "forecasting_engine.services.forecast_enqueuer.forecast_enqueuer.SQS_CLIENT"
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
class TestGetAssetsToForecast:

    def test_only_leaf_assets_returned(
        self, fake_measurement_io, fake_forecast_run_io, fake_assets_io, asset_times
    ):
        leaf_asset_ids = ["asset2", "asset3"]
        fake_assets_io.list_leaf_assets.return_value = [
            MagicMock(asset_uuid=aid) for aid in leaf_asset_ids
        ]
        fake_measurement_io.get_latest_load_per_asset.side_effect = (
            lambda asset_uuids=None: {
                k: v
                for k, v in asset_times.items()
                if k in (asset_uuids or asset_times.keys())
            }
        )
        fake_forecast_run_io.get_latest_forecast_run_start_time_per_asset.return_value = {
            "asset2": asset_times["asset2"][0] - timedelta(minutes=5)
        }

        result = enqueuer.get_assets_to_forecast(
            fake_measurement_io, fake_forecast_run_io, fake_assets_io
        )

        assert "asset1" not in result  # not leaf
        assert "asset2" in result  # leaf, forecast outdated
        assert "asset3" in result  # leaf, no forecast

    def test_no_measurements_returns_empty(
        self, fake_measurement_io, fake_forecast_run_io, fake_assets_io
    ):
        fake_assets_io.list_leaf_assets.return_value = []
        fake_measurement_io.get_latest_load_per_asset.return_value = {}

        result = enqueuer.get_assets_to_forecast(
            fake_measurement_io, fake_forecast_run_io, fake_assets_io
        )

        assert result == {}

    def test_leaf_assets_exist_but_no_measurements(
        self, fake_measurement_io, fake_forecast_run_io, fake_assets_io
    ):
        fake_assets_io.list_leaf_assets.return_value = [MagicMock(asset_uuid="asset1")]

        fake_measurement_io.get_latest_load_per_asset.return_value = {}

        result = enqueuer.get_assets_to_forecast(
            fake_measurement_io, fake_forecast_run_io, fake_assets_io
        )

        assert result == {}

    def test_forecast_newer_than_measurement_skips_asset(
        self, fake_measurement_io, fake_forecast_run_io, fake_assets_io
    ):
        ts_measurement = datetime.now(timezone.utc)
        ts_forecast = ts_measurement + timedelta(minutes=5)

        fake_assets_io.list_leaf_assets.return_value = [MagicMock(asset_uuid="asset1")]

        fake_measurement_io.get_latest_load_per_asset.return_value = {
            "asset1": (ts_measurement, 123.0)
        }

        fake_forecast_run_io.get_latest_forecast_run_start_time_per_asset.return_value = {
            "asset1": ts_forecast
        }

        result = enqueuer.get_assets_to_forecast(
            fake_measurement_io, fake_forecast_run_io, fake_assets_io
        )

        assert result == {}


# ----------------------------
# Tests for enqueue_assets
# ----------------------------
class TestEnqueueAssets:

    def test_sends_sqs_messages(self, fake_sqs_client):
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
class TestEnqueueNewForecasts:

    def test_calls_enqueue_assets(self, monkeypatch, asset_times):
        mock_session = MagicMock()
        monkeypatch.setattr(enqueuer, "SessionLocal", lambda: mock_session)

        fake_measurement_io = MagicMock()
        fake_forecast_run_io = MagicMock()
        monkeypatch.setattr(
            enqueuer, "MeasurementsIO", lambda session: fake_measurement_io
        )
        monkeypatch.setattr(
            enqueuer, "ForecastRunIO", lambda session: fake_forecast_run_io
        )

        fake_assets_to_forecast = {"asset1": asset_times["asset1"][0]}
        monkeypatch.setattr(
            enqueuer, "get_assets_to_forecast", lambda m, f, a: fake_assets_to_forecast
        )

        mock_enqueue_assets = MagicMock()
        monkeypatch.setattr(enqueuer, "enqueue_assets", mock_enqueue_assets)

        enqueuer.enqueue_new_forecasts()
        mock_enqueue_assets.assert_called_once_with(fake_assets_to_forecast)

    def test_logs_when_no_assets(self, monkeypatch):
        mock_session = MagicMock()
        monkeypatch.setattr(enqueuer, "SessionLocal", lambda: mock_session)
        monkeypatch.setattr(enqueuer, "MeasurementsIO", lambda session: MagicMock())
        monkeypatch.setattr(enqueuer, "ForecastRunIO", lambda session: MagicMock())
        monkeypatch.setattr(enqueuer, "get_assets_to_forecast", lambda m, f, a: {})

        with patch(
            "forecasting_engine.services.forecast_enqueuer.forecast_enqueuer.logger"
        ) as mock_logger:
            enqueuer.enqueue_new_forecasts()
            mock_logger.warning.assert_called_with("No assets require a forecast")
