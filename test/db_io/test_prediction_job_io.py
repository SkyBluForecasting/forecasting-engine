import pytest
import pandas as pd
from unittest.mock import MagicMock
from forecasting_engine.db_io.prediction_job_io import PredictionJobIO
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)


def test_get_or_create_returns_existing(make_io, mock_session):
    """Should return ID of existing PredictionJob."""
    io = make_io(PredictionJobIO)

    existing_obj = MagicMock()
    existing_obj.id = 123
    mock_session.query.return_value.filter.return_value.first.return_value = (
        existing_obj
    )

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

    assert result == 123
    mock_session.add.assert_not_called()
    mock_session.commit.assert_not_called()


def test_get_or_create_creates_new(make_io, mock_session):
    """Should create new PredictionJob if not exists."""
    io = make_io(PredictionJobIO)

    # Simulate no existing object
    mock_session.query.return_value.filter.return_value.first.return_value = None

    added_objects = []

    def fake_add(obj):
        obj.id = 456
        added_objects.append(obj)

    mock_session.add.side_effect = fake_add

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

    assert result == 456
    assert len(added_objects) == 1
    added_obj = added_objects[0]
    assert added_obj.asset_uuid == "ASSET2"
    assert added_obj.model == "modelX"
    assert added_obj.horizon_minutes == 120
    assert added_obj.frequency_minutes == 30
    assert added_obj.config["forecast_type"] == "load"
    mock_session.commit.assert_called_once()


def test_get_or_create_raises_on_error(make_io, mock_session):
    """Should rollback and raise if session fails."""
    io = make_io(PredictionJobIO)
    mock_session.query.side_effect = Exception("DB failure")

    pj_dc = PredictionJobDataClass(
        id="ASSET3",
        name="Asset Three",
        model="m",
        horizon_minutes=10,
        resolution_minutes=5,
        quantiles=[0.5],
        forecast_type="wind",
    )

    with pytest.raises(Exception, match="DB failure"):
        io.get_or_create(pj_dc)

    mock_session.rollback.assert_called_once()


def test_to_df_returns_dataframe(make_io, mock_session):
    """Should return a DataFrame with jobs."""
    io = make_io(PredictionJobIO)

    job1 = MagicMock()
    job1.__dict__ = {"id": 1, "asset_uuid": "A", "model": "m"}
    job2 = MagicMock()
    job2.__dict__ = {"id": 2, "asset_uuid": "B", "model": "n"}

    mock_session.query.return_value.all.return_value = [job1, job2]

    df = io.to_df()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert set(df.columns) == {"id", "asset_uuid", "model"}


def test_from_df_passes(make_io, mock_session):
    """Should simply pass (coverage)."""
    io = make_io(PredictionJobIO)
    df = pd.DataFrame()
    result = io.from_df(df)
    assert result is None
