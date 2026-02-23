"""Config file for integration tests."""

import pandas as pd
from forecasting_db.models import WeatherVariable


import os

import pytest
from forecasting_db.models import Base
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from datetime import datetime, timezone
from unittest.mock import MagicMock

from forecasting_db.models import Measurement
from forecasting_engine.db_io.forecast_io import ForecastIO
from forecasting_engine.db_io.forecast_run_io import ForecastRunIO


@pytest.fixture(scope="session")
def test_engine():
    """Create a synchronous SQLAlchemy engine connected to the test database.
    This fixture also ensures required schemas exist and creates all tables
    defined in `Base.metadata`.
    Yields:
        Engine: SQLAlchemy engine connected to the test database.
    """
    TEST_DATABASE_URL = (
        "postgresql://postgres:password@localhost:5432/test_forecasting_db"
    )
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL

    engine = create_engine(TEST_DATABASE_URL, echo=False, future=True)

    # Create schemas
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS openstef;"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS product;"))
        conn.commit()

    # Create tables
    Base.metadata.create_all(bind=engine)

    yield engine
    engine.dispose()


@pytest.fixture
def test_session(test_engine):
    """Provide a synchronous SQLAlchemy session bound to the test engine.
    Args:
        test_engine (Engine): The test database engine fixture.
    Yields:
        Session: SQLAlchemy session.
    """
    SessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    session = SessionLocal()
    try:
        yield session
        session.rollback()  # Rollback any uncommitted changes
    finally:
        session.close()


@pytest.fixture(autouse=True)
def clean_db(test_session):
    """Truncate all tables in the test database before each test.

    Important: tables live in schemas (e.g. openstef.assets), so we must
    truncate fully-qualified names or set a search_path.
    """
    tables = []
    for table in Base.metadata.sorted_tables:
        schema = table.schema
        name = table.name
        if schema:
            tables.append(f'"{schema}"."{name}"')
        else:
            tables.append(f'"{name}"')

    if not tables:
        return

    # TRUNCATE in dependency order (sorted_tables) with CASCADE
    test_session.execute(text(f"TRUNCATE TABLE {', '.join(tables)} CASCADE;"))
    test_session.commit()


@pytest.fixture
def now_utc() -> datetime:
    """Fixed time anchor so timestamps are deterministic."""
    return datetime(2025, 1, 8, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def make_forecast_df():
    """Factory for forecast DataFrames."""

    def _make(
        start: datetime,
        *,
        periods: int,
        resolution_min: int,
        base: float,
        step: float = 0.1,
    ) -> pd.DataFrame:
        ts = pd.date_range(
            start, periods=periods, freq=f"{resolution_min}min", tz="UTC"
        )
        vals = [base + i * step for i in range(len(ts))]
        return pd.DataFrame(
            {
                "timestamp": ts,
                "forecast": vals,
                "p05": [v * 0.9 for v in vals],
                "p95": [v * 1.1 for v in vals],
            }
        )

    return _make


@pytest.fixture
def seed_forecast_run_with_points(test_session):
    """Persist a ForecastRun + Forecast points from a df."""

    def _seed(
        *,
        asset_uuid: str,
        prediction_job_id: int,
        model_run_id: str | None,
        df: pd.DataFrame,
        frequency_min: int,
    ) -> str:
        run_io = ForecastRunIO(test_session)
        fc_io = ForecastIO(test_session)

        run_id = run_io.create(
            prediction_job_id=prediction_job_id,
            model_run_id=model_run_id,
            asset_uuid=asset_uuid,
            start_time=df["timestamp"].iloc[0].to_pydatetime(),
            end_time=df["timestamp"].iloc[-1].to_pydatetime(),
            frequency_min=frequency_min,
        )
        fc_io.from_df(df, run_id)
        test_session.commit()
        return run_id

    return _seed


@pytest.fixture
def insert_measurements(test_session, now_utc):
    """Insert simple hourly load/temp measurements for an asset."""

    def _insert(asset_uuid: str, *, hours: int = 72) -> None:
        ts = pd.date_range(now_utc, periods=hours, freq="h", tz="UTC")
        rows = []
        for t in ts:
            rows.append(
                Measurement(
                    asset_uuid=asset_uuid,
                    timestamp=t.to_pydatetime(),
                    metric="load",
                    value=100.0,
                )
            )
            rows.append(
                Measurement(
                    asset_uuid=asset_uuid,
                    timestamp=t.to_pydatetime(),
                    metric="temp",
                    value=20.0,
                )
            )
        test_session.add_all(rows)
        test_session.commit()

    return _insert


@pytest.fixture
def mock_forecasting_stack(monkeypatch, now_utc, make_forecast_df):
    """Mock MLflowSerializer + pipeline output for deterministic DB integration tests."""
    fake_serializer = MagicMock()
    fake_serializer._find_models.return_value = pd.DataFrame(
        [{"run_id": "MOCK_RUN_123"}]
    )
    fake_model = MagicMock()
    fake_specs = {"feature_names": ["load", "temp"], "hyper_params": {}}
    fake_serializer.load_model.return_value = (fake_model, fake_specs)

    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.MLflowSerializer",
        lambda mlflow_uri: fake_serializer,
    )

    def _mock_create_forecast(pj, prepared_df, model, model_specs):
        periods = pj.horizon_minutes // pj.resolution_minutes
        return make_forecast_df(
            now_utc,
            periods=periods,
            resolution_min=pj.resolution_minutes,
            base=30.0,
            step=0.1,
        )

    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.create_forecast_pipeline_core",
        _mock_create_forecast,
    )
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.normalize_forecast_columns",
        lambda df: df,
    )

    return fake_serializer


# Shared dummy client for deterministic weather data in integration tests
class DummyOpenMeteoClient:
    def fetch_forecast(self, latitude, longitude, hours=None, **kwargs):
        base_time = pd.Timestamp("2026-01-01T00:00:00Z")
        n_hours = hours or kwargs.get("forecast_hours", 6)
        records = []
        for h in range(n_hours):
            for var in WeatherVariable:
                records.append(
                    {
                        "timestamp": base_time + pd.Timedelta(hours=h),
                        "variable": var.value,
                        "value": 10.0 + h,
                        "issue_time": base_time,
                        "weather_site_id": f"cell_{int(latitude*20)}_{int(longitude*20)}",
                    }
                )
        return pd.DataFrame(records)

    def fetch_historical(
        self, latitude, longitude, end_date, start_date=None, **kwargs
    ):
        # Always return the same records regardless of start_date/end_date
        base_time = pd.Timestamp("2025-12-31T18:00:00Z")
        records = []
        for h in range(6):
            for var in WeatherVariable:
                records.append(
                    {
                        "timestamp": base_time + pd.Timedelta(hours=h),
                        "variable": var.value,
                        "value": 5.0 + h,
                        "weather_site_id": f"cell_{int(latitude*20)}_{int(longitude*20)}",
                    }
                )
        return pd.DataFrame(records)
