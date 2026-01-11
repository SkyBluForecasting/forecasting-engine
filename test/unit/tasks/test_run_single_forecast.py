import sys
import pytest
from unittest.mock import patch
import pandas as pd

from forecasting_engine.tasks.run_single_forecast import run_asset_forecast, main


# ----------------------------
# Test run_asset_forecast with various exceptions
# ----------------------------


@pytest.mark.parametrize(
    "generate_return,side_effect,expected_status,expected_msg",
    [
        # generate_forecast returns a df -> success
        (pd.DataFrame({"timestamp": [], "forecast": []}), None, "success", None),
        # generate_forecast returns None -> skipped
        (None, None, "skipped", "No forecast generated"),
        # exceptions -> mapped statuses
        (None, FileNotFoundError("missing"), "not_found", "missing"),
        (None, ValueError("bad input"), "bad_input", "bad input"),
        (None, RuntimeError("fatal"), "fatal_error", "fatal"),
    ],
)
def test_run_asset_forecast_cases(
    generate_return, side_effect, expected_status, expected_msg
):
    with patch(
        "forecasting_engine.shared.forecast_runner.ForecastManager.generate_forecast"
    ) as mock_generate:
        if side_effect is not None:
            mock_generate.side_effect = side_effect
        else:
            mock_generate.return_value = generate_return

        result = run_asset_forecast("asset-x")

        assert result["status"] == expected_status

        if expected_msg is None:
            assert result["message"] is None
        else:
            # message might be longer; just ensure it contains the key text
            assert expected_msg in (result["message"] or "")


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

    # Patch run_asset_forecast in the namespace where main() uses it
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
