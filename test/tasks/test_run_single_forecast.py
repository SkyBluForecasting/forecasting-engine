import sys
import pytest
from unittest.mock import patch

from forecasting_engine.tasks.run_single_forecast import run_asset_forecast, main


# ----------------------------
# Test run_asset_forecast with various exceptions
# ----------------------------
@pytest.mark.parametrize(
    "side_effect,expected_status,expected_msg",
    [
        (None, "success", None),
        (FileNotFoundError("missing"), "not_found", "missing"),
        (ValueError("bad input"), "bad_input", "bad input"),
        (RuntimeError("fatal"), "fatal_error", "fatal"),
    ],
)
def test_run_asset_forecast_cases(side_effect, expected_status, expected_msg):
    # Patch ForecastManager.generate_forecast to simulate success or raise exceptions
    with patch(
        "forecasting_engine.tasks.run_single_forecast.ForecastManager.generate_forecast"
    ) as mock_generate:
        if side_effect is None:
            mock_generate.return_value = None
        else:
            mock_generate.side_effect = side_effect

        result = run_asset_forecast("asset-x")

        # Check status
        assert result["status"] == expected_status

        # Check message contains expected text
        if expected_msg:
            assert expected_msg in result["message"]
        else:
            assert result["message"] is None


# ----------------------------
# Test main() exit codes
# ----------------------------
@pytest.mark.parametrize(
    "status,expected_exit_code",
    [
        ("success", 0),
        ("not_found", 1),
        ("bad_input", 2),
        ("fatal_error", 3),
    ],
)
def test_main_exit_codes(monkeypatch, status, expected_exit_code):
    # Simulate CLI arguments
    monkeypatch.setattr(sys, "argv", ["progname", "ASSET123"])

    fake_result = {"asset_id": "ASSET123", "status": status, "message": None}

    # Patch run_asset_forecast to return a fake result dict
    with patch(
        "forecasting_engine.tasks.run_single_forecast.run_asset_forecast",
        return_value=fake_result,
    ) as mock_run:
        # main() should call sys.exit with the correct code
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == expected_exit_code

        # Verify run_asset_forecast was called once with correct asset ID
        mock_run.assert_called_once_with("ASSET123")
