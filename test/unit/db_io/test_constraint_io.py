import pytest
import pandas as pd

from forecasting_db.models import Constraint, Asset


# ----------------------------
# fixtures
# ----------------------------


@pytest.fixture
def asset(in_memory_session):
    asset = Asset(
        asset_uuid="ASSET1",
        asset_type="system",
        name="Test Asset",
        capacity_kw=100,
        depth=0,
        measured=True,
    )
    in_memory_session.add(asset)
    in_memory_session.flush()
    return asset


@pytest.fixture
def constraint_forecast_df():

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="H"),
            "forecast": [10, 20, 30],
        }
    )


# ----------------------------
# from_forecast tests
# ----------------------------


def test_from_forecast_no_asset_found(
    constraints_io, in_memory_session, constraint_forecast_df
):
    # no Asset inserted on purpose

    constraints_io.from_forecast(
        constraint_forecast_df, forecast_run_id="RUN1", asset_uuid="MISSING"
    )

    assert in_memory_session.query(Constraint).count() == 0


def test_from_forecast_asset_without_capacity(constraints_io, in_memory_session):
    asset = Asset(
        asset_uuid="ASSET1",
        asset_type="system",
        name="Asset",
        capacity_kw=None,
        depth=0,
        measured=True,
    )
    in_memory_session.add(asset)
    in_memory_session.flush()

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "forecast": [10, 20, 30],
        }
    )

    constraints_io.from_forecast(df, "RUN1", asset.asset_uuid)

    assert in_memory_session.query(Constraint).count() == 0


def test_from_forecast_no_violations(constraints_io, in_memory_session, asset):
    asset.capacity_kw = 100
    in_memory_session.flush()

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "forecast": [10, 20, 30],
        }
    )

    constraints_io.from_forecast(df, "RUN1", asset.asset_uuid)

    assert in_memory_session.query(Constraint).count() == 0


def test_from_forecast_with_violations(constraints_io, in_memory_session, asset):
    asset.capacity_kw = 15
    in_memory_session.flush()

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "forecast": [10, 20, 30],
        }
    )

    constraints_io.from_forecast(df, "RUN1", asset.asset_uuid)

    rows = in_memory_session.query(Constraint).order_by(Constraint.timestamp).all()

    assert len(rows) == 2

    assert rows[0].constraint_kw == 5
    assert rows[1].constraint_kw == 15

    assert rows[0].forecast_run_id == "RUN1"
    assert rows[0].asset_uuid == asset.asset_uuid


def test_from_forecast_empty_df(constraints_io, in_memory_session, asset):
    df = pd.DataFrame(columns=["timestamp", "forecast"])

    constraints_io.from_forecast(df, "RUN1", asset.asset_uuid)

    assert in_memory_session.query(Constraint).count() == 0


def test_from_forecast_missing_forecast_column(
    constraints_io, in_memory_session, asset
):
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3, freq="h")})

    with pytest.raises(ValueError):
        constraints_io.from_forecast(df, "RUN1", asset.asset_uuid)

    assert in_memory_session.query(Constraint).count() == 0


# # ----------------------------
# # to_df tests
# # ----------------------------


def test_to_df_returns_dataframe(constraints_io, in_memory_session):
    row1 = Constraint(
        asset_uuid="A1",
        timestamp=pd.Timestamp("2025-01-01T00:00:00"),
        forecast_run_id="RUN1",
        constraint_kw=5.0,
    )
    row2 = Constraint(
        asset_uuid="A1",
        timestamp=pd.Timestamp("2025-01-01T01:00:00"),
        forecast_run_id="RUN1",
        constraint_kw=10.0,
    )
    in_memory_session.add_all([row1, row2])
    in_memory_session.commit()

    df = constraints_io.to_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) == {
        "asset_uuid",
        "timestamp",
        "forecast_run_id",
        "constraint_kw",
    }


def test_to_df_empty_returns_empty_df(constraints_io):
    df = constraints_io.to_df()
    assert df.empty


def test_to_df_filters_by_asset_uuid(constraints_io, in_memory_session):
    row = Constraint(
        asset_uuid="ASSET123",
        timestamp=pd.Timestamp("2025-01-01"),
        forecast_run_id="RUN1",
        constraint_kw=5,
    )
    in_memory_session.add(row)
    in_memory_session.commit()

    df = constraints_io.to_df(asset_uuid="ASSET123")

    assert df["asset_uuid"].tolist() == ["ASSET123"]


def test_from_df_not_implemented(constraints_io):
    df = pd.DataFrame()
    with pytest.raises(NotImplementedError, match="Use `from_forecast`"):
        constraints_io.from_df(df)
