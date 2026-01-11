import sys
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

from forecasting_engine.tasks.run_nonleaf_forecasts import run_bottom_up, main


def _asset(asset_uuid: str):
    """Small helper to make a fake Asset-like object."""
    return SimpleNamespace(asset_uuid=asset_uuid)


def _mock_session_local():
    """Return a SessionLocal mock that yields a mock session."""
    session = MagicMock()
    session.close = MagicMock()
    return session


# ----------------------------
# Ordering: bottom-up by depth
# ----------------------------
def test_run_bottom_up_runs_depths_descending_order():
    session = _mock_session_local()

    depth2_assets = [_asset("A2"), _asset("B2")]
    depth1_assets = [_asset("A1")]
    depth0_assets = [_asset("ROOT")]

    assets_io = MagicMock()
    assets_io.get_max_depth_non_leaf.return_value = 2
    assets_io.list_non_leaf_assets_at_depth.side_effect = [
        depth2_assets,
        depth1_assets,
        depth0_assets,
    ]

    with patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.SessionLocal",
        return_value=session,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.AssetsIO",
        return_value=assets_io,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.run_asset_forecast"
    ) as mock_run:

        mock_run.side_effect = [
            {"status": "success"},
            {"status": "success"},
            {"status": "success"},
            {"status": "success"},
        ]

        result = run_bottom_up(dry_run=False, asset_type=None)

        assert result["status"] == "success"
        assert result["total_attempted"] == 4
        assert result["failures"] == []
        assert result["skipped"] == []

        assert mock_run.call_args_list == [
            call("A2"),
            call("B2"),
            call("A1"),
            call("ROOT"),
        ]


# ----------------------------
# Continues after failures (always)
# ----------------------------
def test_run_bottom_up_continues_after_failures_and_attempts_all_assets():
    session = _mock_session_local()

    depth2_assets = [_asset("A2"), _asset("B2")]
    depth1_assets = [_asset("A1")]
    depth0_assets = [_asset("ROOT")]

    assets_io = MagicMock()
    assets_io.get_max_depth_non_leaf.return_value = 2
    assets_io.list_non_leaf_assets_at_depth.side_effect = [
        depth2_assets,
        depth1_assets,
        depth0_assets,
    ]

    with patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.SessionLocal",
        return_value=session,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.AssetsIO",
        return_value=assets_io,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.run_asset_forecast"
    ) as mock_run:

        # Fail on A2 and A1, succeed others
        mock_run.side_effect = [
            {"status": "fatal_error"},  # A2
            {"status": "success"},  # B2
            {"status": "bad_input"},  # A1
            {"status": "success"},  # ROOT
        ]

        result = run_bottom_up(dry_run=False, asset_type=None)

        assert result["status"] == "fatal_error"
        assert result["total_attempted"] == 4
        assert len(result["failures"]) == 2
        assert result["skipped"] == []

        assert mock_run.call_args_list == [
            call("A2"),
            call("B2"),
            call("A1"),
            call("ROOT"),
        ]

        failure_by_asset = {f["asset_uuid"]: f for f in result["failures"]}
        assert failure_by_asset["A2"]["depth"] == 2
        assert failure_by_asset["A2"]["status"] == "fatal_error"
        assert failure_by_asset["A1"]["depth"] == 1
        assert failure_by_asset["A1"]["status"] == "bad_input"


# ----------------------------
# Skipped behavior: does not count as failure
# ----------------------------
def test_run_bottom_up_tracks_skipped_but_does_not_fail_run():
    session = _mock_session_local()

    depth1_assets = [_asset("A1"), _asset("B1")]

    assets_io = MagicMock()
    assets_io.get_max_depth_non_leaf.return_value = 1
    assets_io.list_non_leaf_assets_at_depth.side_effect = [
        depth1_assets,  # depth=1
        [],  # depth=0
    ]

    with patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.SessionLocal",
        return_value=session,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.AssetsIO",
        return_value=assets_io,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.run_asset_forecast"
    ) as mock_run:

        mock_run.side_effect = [
            {"status": "skipped", "message": "missing child overlap"},
            {"status": "success"},
        ]

        result = run_bottom_up(dry_run=False, asset_type=None)

        # skipped is not a failure, so overall success
        assert result["status"] == "success"
        assert result["total_attempted"] == 2
        assert result["failures"] == []
        assert len(result["skipped"]) == 1
        assert result["skipped"][0]["asset_uuid"] == "A1"
        assert result["skipped"][0]["depth"] == 1
        assert "missing child overlap" in (result["skipped"][0].get("message") or "")

        assert mock_run.call_args_list == [call("A1"), call("B1")]


# ----------------------------
# Dry-run behavior
# ----------------------------
def test_run_bottom_up_dry_run_does_not_call_run_asset_forecast():
    session = _mock_session_local()

    depth2_assets = [_asset("A2")]

    assets_io = MagicMock()
    assets_io.get_max_depth_non_leaf.return_value = 2
    assets_io.list_non_leaf_assets_at_depth.side_effect = [
        depth2_assets,
        [],
        [],
    ]

    with patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.SessionLocal",
        return_value=session,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.AssetsIO",
        return_value=assets_io,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.run_asset_forecast"
    ) as mock_run:

        result = run_bottom_up(dry_run=True, asset_type=None)

        assert result["status"] == "success"
        assert result["total_attempted"] == 1
        assert result["failures"] == []
        assert result["skipped"] == []
        mock_run.assert_not_called()


# ----------------------------
# main() exit codes
# ----------------------------
@pytest.mark.parametrize(
    "fake_result,expected_exit",
    [
        ({"status": "success", "total_attempted": 0, "failures": [], "skipped": []}, 0),
        (
            {
                "status": "fatal_error",
                "total_attempted": 10,
                "failures": [{"asset_uuid": "X", "depth": 1, "status": "fatal_error"}],
                "skipped": [],
            },
            3,
        ),
        # Skips alone should still exit 0 (since status=success)
        (
            {
                "status": "success",
                "total_attempted": 2,
                "failures": [],
                "skipped": [
                    {"asset_uuid": "Y", "depth": 0, "message": "missing overlap"}
                ],
            },
            0,
        ),
    ],
)
def test_main_exit_codes(monkeypatch, fake_result, expected_exit):
    monkeypatch.setattr(sys, "argv", ["progname"])

    with patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.run_bottom_up",
        return_value=fake_result,
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == expected_exit


def test_run_bottom_up_outer_exception_returns_fatal_error_and_closes_session():
    session = _mock_session_local()

    with patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.SessionLocal",
        return_value=session,
    ), patch(
        "forecasting_engine.tasks.run_nonleaf_forecasts.AssetsIO",
        side_effect=Exception("boom"),
    ):

        result = run_bottom_up(dry_run=False, asset_type=None)

        assert result == {"status": "fatal_error"}
        session.close.assert_called_once()
