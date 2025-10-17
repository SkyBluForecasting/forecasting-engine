import pytest
import pandas as pd
from unittest.mock import patch
from forecasting_engine.db_io.forecast_run_io import ForecastRunIO
from forecasting_db.models import ForecastRun


def test_create_success(make_io, mock_session):
    """Should add a new ForecastRun and commit."""
    io = make_io(ForecastRunIO)

    with patch(
        "forecasting_engine.db_io.forecast_run_io.uuid.uuid4", return_value="uuid123"
    ):
        run_id = io.create(
            prediction_job_id=1,
            model_run_id="mlflow123",
            asset_uuid="ASSET1",
        )

    # Verify a ForecastRun was created and added
    args, _ = mock_session.add.call_args
    added_run = args[0]
    assert isinstance(added_run, ForecastRun)
    assert added_run.forecast_run_id == "uuid123"
    assert added_run.prediction_job_id == 1
    assert added_run.model_run_id == "mlflow123"
    assert added_run.asset_uuid == "ASSET1"

    mock_session.commit.assert_called_once()
    assert run_id == "uuid123"


def test_create_rollback_on_error(make_io, mock_session):
    """Should rollback and raise on error."""
    io = make_io(ForecastRunIO)
    mock_session.add.side_effect = Exception("DB failure")

    with pytest.raises(Exception, match="DB failure"):
        io.create(1, "mlflow123", "ASSET1")

    mock_session.rollback.assert_called_once()


def test_update_status_updates_fields(make_io, mock_session):
    """Should update given fields and commit."""
    io = make_io(ForecastRunIO)
    mock_run = ForecastRun()
    mock_session.get.return_value = mock_run

    io.update_status("RUN1", frequency_min=60)

    assert mock_run.frequency_min == 60
    mock_session.commit.assert_called_once()


def test_update_status_skips_missing_attributes(make_io, mock_session):
    """Should update valid fields and skip invalid ones."""
    io = make_io(ForecastRunIO)
    mock_run = type("MockRun", (), {"frequency_min": None})()  # has only frequency_min
    mock_session.get.return_value = mock_run

    io.update_status("RUN1", frequency_min=30, not_a_field="X")

    # ✅ only the valid attribute updated
    assert mock_run.frequency_min == 30
    # ✅ invalid attribute was ignored
    assert not hasattr(mock_run, "not_a_field")
    mock_session.commit.assert_called_once()


def test_update_status_no_run_found(make_io, mock_session):
    """Should do nothing if no ForecastRun found."""
    io = make_io(ForecastRunIO)
    mock_session.get.return_value = None

    io.update_status("RUN1", status="FAIL")
    mock_session.commit.assert_not_called()


def test_to_df_returns_dataframe(make_io, mock_session):
    """Should return DataFrame with cleaned columns."""

    class MockRun:
        def __init__(self):
            self.forecast_run_id = "RUN1"
            self.prediction_job_id = 1
            self.model_run_id = "MLFLOW"
            self.asset_uuid = "ASSET1"

    mock_session.query.return_value.all.return_value = [MockRun()]

    io = make_io(ForecastRunIO)
    df = io.to_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "forecast_run_id" in df.columns


def test_from_df_noop(make_io, mock_session):
    """from_df should be a no-op (placeholder)."""
    io = make_io(ForecastRunIO)
    df = pd.DataFrame()
    result = io.from_df(df)
    assert result is None
