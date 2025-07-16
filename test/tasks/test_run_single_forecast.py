import pytest
from unittest.mock import patch
from forecasting_engine.tasks.run_single_forecast import run_asset_forecast


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
