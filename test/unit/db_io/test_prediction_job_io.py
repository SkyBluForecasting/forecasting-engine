import pytest
import pandas as pd
from forecasting_engine.db_io.prediction_job_io import PredictionJobIO
from forecasting_db.models import PredictionJob
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)


# ----------------- get_or_create tests -----------------


def test_get_or_create_returns_existing(in_memory_session):
    """Should return ID of existing PredictionJob without creating new."""
    io = PredictionJobIO(in_memory_session)

    # Insert an existing job
    existing_job = PredictionJob(
        asset_uuid="ASSET1",
        model="my_model",
        horizon_minutes=60,
        frequency_minutes=15,
        config={"forecast_type": "solar"},
    )
    in_memory_session.add(existing_job)
    in_memory_session.commit()

    pj_dc = PredictionJobDataClass(
        id="ASSET1",
        name="Asset One",
        model="my_model",
        horizon_minutes=60,
        resolution_minutes=15,
        quantiles=[0.05, 0.5, 0.95],
        forecast_type="solar",
    )

    result = io.get_or_create(pj_dc)

    assert result == existing_job.id
    # No new job should be added
    all_jobs = in_memory_session.query(PredictionJob).all()
    assert len(all_jobs) == 1


def test_get_or_create_creates_new(in_memory_session):
    """Should create a new PredictionJob if not exists."""
    io = PredictionJobIO(in_memory_session)

    pj_dc = PredictionJobDataClass(
        id="ASSET2",
        name="Asset Two",
        model="modelX",
        horizon_minutes=120,
        resolution_minutes=30,
        quantiles=[0.1, 0.5, 0.9],
        forecast_type="load",
    )

    result = io.get_or_create(pj_dc)

    # Verify new job is created
    new_job = (
        in_memory_session.query(PredictionJob).filter_by(asset_uuid="ASSET2").first()
    )
    assert result == new_job.id
    assert new_job.model == "modelX"
    assert new_job.horizon_minutes == 120
    assert new_job.frequency_minutes == 30
    assert new_job.config["forecast_type"] == "load"


def test_get_or_create_raises_and_rolls_back(in_memory_session):
    """Should raise an exception and rollback if session fails during get_or_create."""
    io = PredictionJobIO(in_memory_session)

    pj_dc = PredictionJobDataClass(
        id="ASSET_ERROR",
        name="ASSET ERROR",
        model="modelX",
        horizon_minutes=60,
        resolution_minutes=15,
        quantiles=[0.1, 0.5, 0.9],
        forecast_type="solar",
    )

    # Patch session.commit to force an error
    original_commit = in_memory_session.commit

    def fail_commit():
        raise RuntimeError("Forced commit failure")

    in_memory_session.commit = fail_commit

    with pytest.raises(
        RuntimeError, match="Failed to get/create PredictionJob for asset ASSET_ERROR"
    ):
        io.get_or_create(pj_dc)

    # Restore commit method
    in_memory_session.commit = original_commit

    # DB should remain empty
    jobs = in_memory_session.query(PredictionJob).all()
    assert len(jobs) == 0


# ----------------- to_df / from_df tests -----------------


def test_to_df_returns_dataframe(in_memory_session):
    """to_df should return DataFrame of prediction jobs."""
    io = PredictionJobIO(in_memory_session)

    # Insert two jobs
    jobs = [
        PredictionJob(
            asset_uuid="A",
            model="m",
            horizon_minutes=60,
            frequency_minutes=15,
            config={},
        ),
        PredictionJob(
            asset_uuid="B",
            model="n",
            horizon_minutes=120,
            frequency_minutes=30,
            config={},
        ),
    ]
    in_memory_session.add_all(jobs)
    in_memory_session.commit()

    df = io.to_df()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert set(df["asset_uuid"]) == {"A", "B"}
    assert set(df["model"]) == {"m", "n"}


def test_from_df_passes(in_memory_session):
    """from_df is a no-op for PredictionJobIO (coverage)."""
    io = PredictionJobIO(in_memory_session)
    df = pd.DataFrame()
    result = io.from_df(df)
    assert result is None
