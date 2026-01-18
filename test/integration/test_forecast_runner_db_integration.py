"""DB integration tests for ForecastManager + db, using a real Postgres test DB.

Note:
- MLflow + OpenSTEF pipeline are mocked to keep these tests deterministic and
  focused on DB persistence + orchestration.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from forecasting_db.models import Asset, ForecastRun, Forecast, PredictionJob
from forecasting_engine.shared.forecast_runner import ForecastManager
from forecasting_engine.db_io.forecast_io import ForecastIO
from forecasting_engine.config import DEFAULT_FORECAST_HORIZON_MINUTES
from forecasting_engine.db_io.prediction_job_io import PredictionJobIO
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)


# ----------------------------
# DB fixtures: assets
# ----------------------------


@pytest.fixture
def leaf_asset(test_session) -> Asset:
    asset = Asset(
        asset_uuid="LEAF1",
        asset_type="distribution_substation",
        name="Leaf Asset 1",
        depth=0,
        measured=True,
        capacity_kw=100.0,
    )
    test_session.add(asset)
    test_session.commit()
    return asset


@pytest.fixture
def parent_asset(test_session) -> Asset:
    asset = Asset(
        asset_uuid="PARENT1",
        asset_type="system",
        name="Parent Asset",
        depth=1,
        measured=False,
        capacity_kw=200.0,
    )
    test_session.add(asset)
    test_session.commit()
    return asset


@pytest.fixture
def child_assets(test_session, parent_asset) -> list[Asset]:
    child1 = Asset(
        asset_uuid="CHILD1",
        asset_type="distribution_substation",
        name="Child 1",
        depth=0,
        measured=True,
        parent_uuid=parent_asset.asset_uuid,
        capacity_kw=50.0,
    )
    child2 = Asset(
        asset_uuid="CHILD2",
        asset_type="distribution_substation",
        name="Child 2",
        depth=0,
        measured=True,
        parent_uuid=parent_asset.asset_uuid,
        capacity_kw=75.0,
    )
    test_session.add_all([child1, child2])
    test_session.commit()
    return [child1, child2]


# ----------------------------
# Tests
# ----------------------------


def test_model_forecast_persists_run_points_and_roundtrips(
    test_session,
    leaf_asset,
    insert_measurements,
    mock_forecasting_stack,
):
    insert_measurements(leaf_asset.asset_uuid)

    df = ForecastManager(test_session).generate_forecast(leaf_asset.asset_uuid)

    assert df is not None and not df.empty
    assert {"timestamp", "forecast"}.issubset(df.columns)

    run = (
        test_session.query(ForecastRun)
        .filter(ForecastRun.asset_uuid == leaf_asset.asset_uuid)
        .one()
    )
    assert run.model_run_id == "MOCK_RUN_123"

    points = (
        test_session.query(Forecast)
        .filter(Forecast.forecast_run_id == run.forecast_run_id)
        .all()
    )
    assert len(points) == len(df)

    pj = (
        test_session.query(PredictionJob)
        .filter(PredictionJob.id == run.prediction_job_id)
        .one()
    )
    assert pj.asset_uuid == leaf_asset.asset_uuid

    retrieved = ForecastIO(test_session).to_df(run.forecast_run_id)
    assert len(retrieved) == len(df)

    merged = df.merge(retrieved, on="timestamp", suffixes=("_orig", "_retr"))
    assert (merged["forecast_orig"] - merged["forecast_retr"]).abs().max() < 0.01


def test_parent_aggregates_existing_child_forecasts_and_persists(
    test_session,
    parent_asset,
    child_assets,
    make_forecast_df,
    seed_forecast_run_with_points,
):
    pj_io = PredictionJobIO(test_session)
    seed_now = datetime.now(timezone.utc)

    def seed_child(child_uuid: str, model_run_id: str, base: float):
        pj = PredictionJobDataClass(
            id=child_uuid,
            model="xgb",
            quantiles=[0.05, 0.95],
            forecast_type="demand",
            lat=52.0,
            lon=5.0,
            horizon_minutes=DEFAULT_FORECAST_HORIZON_MINUTES,
            resolution_minutes=60,
            name=child_uuid,
            hyper_params={},
            feature_names=None,
            default_modelspecs=None,
            save_train_forecasts=True,
        )
        pj_id = pj_io.get_or_create(pj)

        df = make_forecast_df(
            seed_now, periods=48, resolution_min=60, base=base, step=0.1
        )
        seed_forecast_run_with_points(
            asset_uuid=child_uuid,
            prediction_job_id=pj_id,
            model_run_id=model_run_id,
            df=df,
            frequency_min=60,
        )

    seed_child(child_assets[0].asset_uuid, "CHILD1_RUN", base=40.0)
    seed_child(child_assets[1].asset_uuid, "CHILD2_RUN", base=60.0)

    parent_df = ForecastManager(test_session).generate_forecast(parent_asset.asset_uuid)

    assert parent_df is not None and not parent_df.empty

    expected = np.array([100.0 + i * 0.2 for i in range(len(parent_df))], dtype=float)
    np.testing.assert_allclose(
        parent_df["forecast"].to_numpy(dtype=float), expected, rtol=0, atol=1e-9
    )

    parent_run = (
        test_session.query(ForecastRun)
        .filter(ForecastRun.asset_uuid == parent_asset.asset_uuid)
        .one()
    )
    assert parent_run.model_run_id is None

    parent_points = (
        test_session.query(Forecast)
        .filter(Forecast.forecast_run_id == parent_run.forecast_run_id)
        .all()
    )
    assert len(parent_points) == len(parent_df)
