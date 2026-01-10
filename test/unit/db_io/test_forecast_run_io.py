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


def test_get_latest_forecast_run_start_time_per_asset_returns_empty_if_no_assets(
    in_memory_session,
):
    """Should return empty dict if no asset_uuids provided."""
    io = ForecastRunIO(in_memory_session)
    result = io.get_latest_forecast_run_start_time_per_asset([])
    assert result == {}


def test_get_latest_forecast_run_start_time_per_asset_returns_correct_mapping(
    in_memory_session,
):
    """Should return dict of asset_uuid -> latest start_time."""
    io = ForecastRunIO(in_memory_session)

    # Create two runs with different start_times
    run1 = io.create(1, "mlflow123", "A1")
    run2 = io.create(2, "mlflow456", "A2")

    # Manually set start_times
    io.update_status(run1, start_time=datetime.datetime(2025, 1, 1, 12, 0))
    io.update_status(run2, start_time=datetime.datetime(2025, 1, 1, 11, 0))

    result = io.get_latest_forecast_run_start_time_per_asset(["A1", "A2"])
    assert result == {
        "A1": datetime.datetime(2025, 1, 1, 12, 0),
        "A2": datetime.datetime(2025, 1, 1, 11, 0),
    }


# ============================
# get_latest_overlapping_forecast_run_id
# ============================


def test_get_latest_overlapping_forecast_run_id_returns_none_if_no_runs(
    in_memory_session,
):
    io = ForecastRunIO(in_memory_session)

    got = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 0),
        window_end=datetime.datetime(2025, 1, 1, 11, 0),
    )
    assert got is None


def test_get_latest_overlapping_forecast_run_id_returns_none_if_no_overlap(
    in_memory_session,
):
    io = ForecastRunIO(in_memory_session)

    run_id = io.create(1, "mlflow123", "A1")
    io.update_status(
        run_id,
        start_time=datetime.datetime(2025, 1, 1, 8, 0),
        end_time=datetime.datetime(2025, 1, 1, 9, 0),
    )

    got = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 0),
        window_end=datetime.datetime(2025, 1, 1, 11, 0),
    )
    assert got is None


def test_get_latest_overlapping_forecast_run_id_overlap_interior(in_memory_session):
    io = ForecastRunIO(in_memory_session)

    run_id = io.create(1, "mlflow123", "A1")
    io.update_status(
        run_id,
        start_time=datetime.datetime(2025, 1, 1, 9, 0),
        end_time=datetime.datetime(2025, 1, 1, 12, 0),
    )

    got = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 0),
        window_end=datetime.datetime(2025, 1, 1, 11, 0),
    )
    assert got == run_id


def test_get_latest_overlapping_forecast_run_id_overlap_on_boundaries(
    in_memory_session,
):
    io = ForecastRunIO(in_memory_session)

    # run ends exactly at window_start -> overlaps (end_time >= window_start)
    run_id_1 = io.create(1, "mlflow123", "A1")
    io.update_status(
        run_id_1,
        start_time=datetime.datetime(2025, 1, 1, 8, 0),
        end_time=datetime.datetime(2025, 1, 1, 10, 0),
    )

    got1 = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 0),
        window_end=datetime.datetime(2025, 1, 1, 11, 0),
    )
    assert got1 == run_id_1

    # run starts exactly at window_end -> overlaps (start_time <= window_end)
    run_id_2 = io.create(2, "mlflow456", "A1")
    io.update_status(
        run_id_2,
        start_time=datetime.datetime(2025, 1, 1, 11, 0),
        end_time=datetime.datetime(2025, 1, 1, 12, 0),
    )

    got2 = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 0),
        window_end=datetime.datetime(2025, 1, 1, 11, 0),
    )
    # should pick the latest overlapping by start_time (11:00)
    assert got2 == run_id_2


def test_get_latest_overlapping_forecast_run_id_picks_latest_by_start_time(
    in_memory_session,
):
    io = ForecastRunIO(in_memory_session)

    # Older overlapping run
    old_id = io.create(1, "mlflow_old", "A1")
    io.update_status(
        old_id,
        start_time=datetime.datetime(2025, 1, 1, 9, 0),
        end_time=datetime.datetime(2025, 1, 1, 13, 0),
    )

    # Newer overlapping run
    new_id = io.create(2, "mlflow_new", "A1")
    io.update_status(
        new_id,
        start_time=datetime.datetime(2025, 1, 1, 11, 0),
        end_time=datetime.datetime(2025, 1, 1, 12, 0),
    )

    got = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 30),
        window_end=datetime.datetime(2025, 1, 1, 11, 30),
    )
    assert got == new_id


def test_get_latest_overlapping_forecast_run_id_ignores_other_assets(in_memory_session):
    io = ForecastRunIO(in_memory_session)

    a1_id = io.create(1, "mlflow_a1", "A1")
    io.update_status(
        a1_id,
        start_time=datetime.datetime(2025, 1, 1, 9, 0),
        end_time=datetime.datetime(2025, 1, 1, 12, 0),
    )

    a2_id = io.create(2, "mlflow_a2", "A2")
    io.update_status(
        a2_id,
        start_time=datetime.datetime(2025, 1, 1, 11, 0),
        end_time=datetime.datetime(2025, 1, 1, 13, 0),
    )

    got = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 0),
        window_end=datetime.datetime(2025, 1, 1, 11, 0),
    )
    assert got == a1_id


def test_get_latest_overlapping_forecast_run_id_end_time_null_is_open_ended(
    in_memory_session,
):
    io = ForecastRunIO(in_memory_session)

    open_id = io.create(1, "mlflow_open", "A1")
    io.update_status(
        open_id,
        start_time=datetime.datetime(2025, 1, 1, 9, 0),
        end_time=None,
    )

    got = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 0),
        window_end=datetime.datetime(2025, 1, 1, 11, 0),
    )
    assert got == open_id


def test_get_latest_overlapping_forecast_run_id_null_end_time_still_needs_start_before_window_end(
    in_memory_session,
):
    io = ForecastRunIO(in_memory_session)

    open_id = io.create(1, "mlflow_open", "A1")
    io.update_status(
        open_id,
        start_time=datetime.datetime(2025, 1, 1, 12, 0),  # starts after window_end
        end_time=None,
    )

    got = io.get_latest_overlapping_forecast_run_id(
        asset_uuid="A1",
        window_start=datetime.datetime(2025, 1, 1, 10, 0),
        window_end=datetime.datetime(2025, 1, 1, 11, 0),
    )
    assert got is None
