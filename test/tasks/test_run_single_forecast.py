import pytest
import sys
from unittest.mock import patch
from forecasting_engine.tasks.run_single_forecast import run_asset_forecast, main


@pytest.fixture
def mock_generate():
    with patch(
        "forecasting_engine.tasks.run_single_forecast.generate_forecast_for_asset"
    ) as mock:
        yield mock


@pytest.mark.parametrize(
    "side_effect,expected_status,expected_msg",
    [
        (None, "success", None),
        (FileNotFoundError("missing"), "not_found", "missing"),
        (ValueError("bad input"), "bad_input", "bad input"),
        (RuntimeError("fatal"), "fatal_error", "fatal"),
    ],
)
def test_run_asset_forecast_cases(
    mock_generate, side_effect, expected_status, expected_msg
):
    mock_generate.side_effect = side_effect
    result = run_asset_forecast("asset-x")
    assert result["status"] == expected_status
    if expected_msg:
        assert expected_msg in result["message"]
    else:
        assert result["message"] is None


def test_main_runs_forecast_and_exits(monkeypatch):
    # Simulate CLI arguments: script name + asset_id
    monkeypatch.setattr(sys, "argv", ["progname", "ASSET123"])

    # Patch run_asset_forecast to return 0 without doing real work
    with patch(
        "forecasting_engine.tasks.run_single_forecast.run_asset_forecast",
        return_value=0,
    ) as mock_run:
        with pytest.raises(SystemExit) as excinfo:
            main()

        # sys.exit should be called with the return value from run_asset_forecast
        assert excinfo.value.code == 0

        # run_asset_forecast should be called once with the asset ID
        mock_run.assert_called_once_with("ASSET123")
