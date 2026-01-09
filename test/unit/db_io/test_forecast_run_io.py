import pytest
import uuid
import pandas as pd
from forecasting_engine.db_io.forecast_run_io import ForecastRunIO
from forecasting_db.models import ForecastRun
import datetime


def test_create_success(in_memory_session):
    """Should add a new ForecastRun and commit."""
    io = ForecastRunIO(in_memory_session)

    # Patch uuid4 to produce predictable ID
    orig_uuid4 = uuid.uuid4
    uuid.uuid4 = lambda: "uuid123"

    run_id = io.create(
        prediction_job_id=1,
        model_run_id="mlflow123",
        asset_uuid="ASSET1",
    )

    uuid.uuid4 = orig_uuid4  # restore

    # Verify ForecastRun was created
    created = in_memory_session.query(ForecastRun).get(run_id)
    assert isinstance(created, ForecastRun)
    assert created.forecast_run_id == "uuid123"
    assert created.prediction_job_id == 1
    assert created.model_run_id == "mlflow123"
    assert created.asset_uuid == "ASSET1"


def test_create_rollback_on_error(in_memory_session):
    """Should rollback and raise on error."""
    io = ForecastRunIO(in_memory_session)

    # Monkeypatch add to raise error
    orig_add = in_memory_session.add

    def fail_add(x):
        raise Exception("DB failure")

    in_memory_session.add = fail_add

    with pytest.raises(Exception, match="DB failure"):
        io.create(1, "mlflow123", "ASSET1")

    in_memory_session.add = orig_add  # restore


def test_update_status_updates_fields(in_memory_session):
    """Should update given fields and commit."""
    io = ForecastRunIO(in_memory_session)
    run = io.create(1, "mlflow123", "ASSET1")

    io.update_status(run, frequency_min=60)

    updated = in_memory_session.query(ForecastRun).get(run)
    assert updated.frequency_min == 60


def test_update_status_skips_missing_attributes(in_memory_session):
    """Should update valid fields and skip invalid ones."""
    io = ForecastRunIO(in_memory_session)
    run = io.create(1, "mlflow123", "ASSET1")

    # Pass an invalid attribute
    io.update_status(run, frequency_min=30, not_a_field="X")

    updated = in_memory_session.query(ForecastRun).get(run)
    assert updated.frequency_min == 30
    # invalid field should not exist
    assert not hasattr(updated, "not_a_field")


def test_update_status_no_run_found(in_memory_session):
    """Should do nothing if no ForecastRun found."""
    io = ForecastRunIO(in_memory_session)
    # Random ID that doesn't exist
    io.update_status("NONEXISTENT", status="FAIL")
    # Nothing should raise, nothing to assert beyond no exception


def test_to_df_returns_dataframe(in_memory_session):
    """Should return DataFrame with forecast_run_id column."""
    io = ForecastRunIO(in_memory_session)
    io.create(1, "mlflow123", "ASSET1")

    df = io.to_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "forecast_run_id" in df.columns


def test_from_df_noop(in_memory_session):
    """from_df should be a no-op (placeholder)."""
    io = ForecastRunIO(in_memory_session)
    df = pd.DataFrame()
    result = io.from_df(df)
    assert result is None


def test_get_latest_forecast_run_per_asset_returns_empty_if_no_assets(
    in_memory_session,
):
    """Should return empty dict if no asset_uuids provided."""
    io = ForecastRunIO(in_memory_session)
    result = io.get_latest_forecast_run_per_asset([])
    assert result == {}


def test_get_latest_forecast_run_per_asset_returns_correct_mapping(in_memory_session):
    """Should return dict of asset_uuid -> latest start_time."""
    io = ForecastRunIO(in_memory_session)

    # Create two runs with different start_times
    run1 = io.create(1, "mlflow123", "A1")
    run2 = io.create(2, "mlflow456", "A2")

    # Manually set start_times
    io.update_status(run1, start_time=datetime.datetime(2025, 1, 1, 12, 0))
    io.update_status(run2, start_time=datetime.datetime(2025, 1, 1, 11, 0))

    result = io.get_latest_forecast_run_per_asset(["A1", "A2"])
    assert result == {
        "A1": datetime.datetime(2025, 1, 1, 12, 0),
        "A2": datetime.datetime(2025, 1, 1, 11, 0),
    }
