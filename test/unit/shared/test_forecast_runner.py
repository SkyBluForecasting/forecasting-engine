import pytest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime, timezone
from forecasting_engine.config import DEFAULT_FORECAST_HORIZON_MINUTES
from forecasting_engine.shared.forecast_runner import (
    ForecastManager,
    run_asset_forecast,
    _run_asset_forecast_inner,
)


# ============================
# Fixtures
# ============================


@pytest.fixture
def fm():
    """ForecastManager with a mocked session."""
    return ForecastManager(session=MagicMock())


@pytest.fixture
def asset_factory():
    """Factory for creating asset mocks with required attributes."""

    def _make(asset_uuid="A1", children=None, measured=True):
        asset = MagicMock()
        asset.asset_uuid = asset_uuid
        asset.id = asset_uuid  # used in logging
        asset.children = children or []
        asset.measured = measured
        return asset

    return _make


@pytest.fixture
def attach_asset_getter(fm):
    """Attach assets_io.get_asset to return a given asset."""
    fm.assets_io = MagicMock()

    def _attach(asset):
        fm.assets_io.get_asset.return_value = asset
        return asset

    return _attach


@pytest.fixture
def patch_mlflow_success(monkeypatch):
    """
    Patch MLflowSerializer + pipeline for a happy-path model-based forecast.
    Returns the fake serializer so tests can assert calls.
    """
    fake_serializer = MagicMock()
    fake_serializer._find_models.return_value = pd.DataFrame([{"run_id": "RUN123"}])
    fake_serializer.load_model.return_value = ("model", {"spec": "dummy"})

    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.MLflowSerializer",
        lambda _: fake_serializer,
    )
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.normalize_forecast_columns",
        lambda df: df,
    )
    return fake_serializer


@pytest.fixture
def patch_constraints_io(monkeypatch):
    """Patch ConstraintsIO(session) constructor to return a mock instance."""
    fake_constraints = MagicMock()
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.ConstraintsIO",
        lambda _: fake_constraints,
    )
    return fake_constraints


# ============================
# Helpers
# ============================


def make_dummy_measurements():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            "load": [10, 20, 30],
        }
    )


def df_child(start: str, periods: int = 3):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq="h", tz="UTC"),
            "forecast": list(range(1, periods + 1)),
        }
    )


def now_utc():
    return datetime.now(timezone.utc)


def make_parent_with_children(asset_factory, *child_ids):
    return asset_factory(
        asset_uuid="PARENT",
        children=[asset_factory(cid) for cid in child_ids],
    )


# ============================
# _run_model_based_forecast
# ============================


def test_run_model_based_forecast_success(monkeypatch, fm, patch_mlflow_success):
    asset = MagicMock(asset_uuid="ASSET123", measured=True)

    monkeypatch.setattr(fm, "_load_measurements", lambda _: make_dummy_measurements())
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.create_forecast_pipeline_core",
        lambda *args: df_child("2025-01-01"),
    )

    result = fm._run_model_based_forecast(asset)

    assert result.model_run_id == "RUN123"
    assert not result.forecast_df.empty
    patch_mlflow_success._find_models.assert_called_once()
    patch_mlflow_success.load_model.assert_called_once()


def test_run_model_based_forecast_no_model(monkeypatch, fm):
    asset = MagicMock(asset_uuid="ASSET123", measured=True)

    monkeypatch.setattr(fm, "_load_measurements", lambda _: make_dummy_measurements())
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.MLflowSerializer",
        lambda _: MagicMock(_find_models=MagicMock(return_value=pd.DataFrame())),
    )

    with pytest.raises(LookupError, match="No model found"):
        fm._run_model_based_forecast(asset)


# ============================
# run_asset_forecast wrapper
# ============================


def test_run_asset_forecast_success(monkeypatch):
    mock_fm = MagicMock()
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.ForecastManager",
        MagicMock(return_value=mock_fm),
    )

    result = run_asset_forecast("ASSET123")

    mock_fm.generate_forecast.assert_called_once_with("ASSET123")
    assert result["status"] == "success"


@pytest.mark.parametrize(
    "exception,status",
    [
        (ValueError("bad"), "bad_input"),
        (FileNotFoundError("missing"), "not_found"),
        (RuntimeError("boom"), "fatal_error"),
    ],
)
def test_run_asset_forecast_exceptions(monkeypatch, exception, status):
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner._run_asset_forecast_inner",
        lambda _: (_ for _ in ()).throw(exception),
    )

    result = run_asset_forecast("ASSET123")
    assert result["status"] == status


def test_run_asset_forecast_inner(monkeypatch):
    session = MagicMock()
    session.__enter__.return_value = session

    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.SessionLocal",
        lambda: session,
    )
    fm_class = MagicMock()
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.ForecastManager",
        fm_class,
    )

    result = _run_asset_forecast_inner("ASSET123")

    fm_class.return_value.generate_forecast.assert_called_once_with("ASSET123")
    assert result["status"] == "success"


# ============================
# generate_forecast branching
# ============================


def test_generate_forecast_leaf_measured_false(
    monkeypatch, fm, asset_factory, attach_asset_getter
):
    attach_asset_getter(asset_factory(measured=False))

    monkeypatch.setattr(fm, "_run_model_based_forecast", MagicMock())
    monkeypatch.setattr(fm, "_save_forecast_results", MagicMock())

    assert fm.generate_forecast("A1") is None
    fm._run_model_based_forecast.assert_not_called()


def test_generate_forecast_leaf_measured_true(
    monkeypatch, fm, asset_factory, attach_asset_getter
):
    asset = attach_asset_getter(asset_factory(measured=True))

    fake_result = MagicMock(
        forecast_df=df_child("2025-01-01"),
        prediction_job=MagicMock(),
        model_run_id="RUNID",
    )

    monkeypatch.setattr(
        fm, "_run_model_based_forecast", MagicMock(return_value=fake_result)
    )
    save_spy = MagicMock()
    monkeypatch.setattr(fm, "_save_forecast_results", save_spy)

    out = fm.generate_forecast("A1")

    assert not out.empty
    fm._run_model_based_forecast.assert_called_once_with(asset)
    save_spy.assert_called_once()


def test_generate_forecast_parent(monkeypatch, fm, asset_factory, attach_asset_getter):
    parent = attach_asset_getter(
        asset_factory(asset_uuid="PARENT", children=[asset_factory("CH1")])
    )

    fake_result = MagicMock(
        forecast_df=df_child("2025-01-01"),
        prediction_job=MagicMock(),
        model_run_id=None,
    )

    monkeypatch.setattr(
        fm, "_run_hierarchy_based_forecast", MagicMock(return_value=fake_result)
    )
    save_spy = MagicMock()
    monkeypatch.setattr(fm, "_save_forecast_results", save_spy)

    out = fm.generate_forecast("PARENT")

    assert not out.empty
    fm._run_hierarchy_based_forecast.assert_called_once_with(parent)
    save_spy.assert_called_once()


def test_generate_forecast_returns_none_if_model_runner_returns_none(
    monkeypatch, fm, asset_factory, attach_asset_getter
):
    asset = attach_asset_getter(
        asset_factory(asset_uuid="A1", children=[], measured=True)
    )

    monkeypatch.setattr(fm, "_run_model_based_forecast", MagicMock(return_value=None))
    save_spy = MagicMock()
    monkeypatch.setattr(fm, "_save_forecast_results", save_spy)

    out = fm.generate_forecast("A1")

    assert out is None
    fm._run_model_based_forecast.assert_called_once_with(asset)
    save_spy.assert_not_called()


# ============================
# _load_measurements
# ============================


def test_load_measurements_returns_df_when_present(fm):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "load": [10, 20],
        }
    )

    fm.measurements_io = MagicMock()
    fm.measurements_io.to_df.return_value = df

    out = fm._load_measurements("A1")

    assert out is df
    fm.measurements_io.to_df.assert_called_once_with("A1")


def test_load_measurements_raises_if_empty(fm):
    fm.measurements_io = MagicMock()
    fm.measurements_io.to_df.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="No measurements found"):
        fm._load_measurements("A1")


# ============================
# _get_latest_forecast_df
# ============================


def test_get_latest_forecast_df_returns_none_if_no_run(fm):
    fm.forecast_run_io = MagicMock()
    fm.forecast_run_io.get_latest_overlapping_forecast_run_id.return_value = None

    out = fm._get_latest_forecast_df("CH1", now_utc(), now_utc())
    assert out is None


def test_get_latest_forecast_df_returns_none_if_empty_df(fm):
    fm.forecast_run_io = MagicMock()
    fm.forecast_run_io.get_latest_overlapping_forecast_run_id.return_value = "RID"
    fm.forecast_io = MagicMock()
    fm.forecast_io.to_df.return_value = pd.DataFrame()

    out = fm._get_latest_forecast_df("CH1", now_utc(), now_utc())
    assert out is None


def test_get_latest_forecast_df_sorts(fm):
    fm.forecast_run_io = MagicMock()
    fm.forecast_run_io.get_latest_overlapping_forecast_run_id.return_value = "RID"
    fm.forecast_io = MagicMock()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01T01:00:00Z", "2025-01-01T00:00:00Z"], utc=True
            ),
            "forecast": [2, 1],
        }
    )
    fm.forecast_io.to_df.return_value = df

    out = fm._get_latest_forecast_df("CH1", now_utc(), now_utc())
    assert list(out["forecast"]) == [1, 2]


# ============================
# _aggregate_child_forecasts
# ============================


def test_aggregate_child_forecasts_sums(fm):
    a = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "CH1": [1.0, 2.0],
        }
    )
    b = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "CH2": [10.0, 20.0],
        }
    )

    out = fm._aggregate_child_forecasts([a, b])
    assert list(out["forecast"]) == [11.0, 22.0]


def test_aggregate_child_forecasts_empty_merge(fm):
    a = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "CH1": [1.0, 2.0],
        }
    )
    b = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-02", periods=2, freq="h", tz="UTC"),
            "CH2": [10.0, 20.0],
        }
    )

    out = fm._aggregate_child_forecasts([a, b])
    assert out.empty
    assert list(out.columns) == ["timestamp", "forecast"]


# ============================
# _fetch_latest_child_forecasts_in_timerange
# ============================


def test_fetch_latest_child_forecasts_uses_common_window(
    monkeypatch, fm, asset_factory
):
    parent = make_parent_with_children(asset_factory, "CH1")

    child_full = df_child("2025-01-01", periods=4)  # 00:00..03:00
    monkeypatch.setattr(
        fm, "_get_latest_forecast_df", MagicMock(return_value=child_full)
    )

    target_start = pd.Timestamp("2026-01-10T00:00:00Z").to_pydatetime()
    target_end = pd.Timestamp("2026-01-12T00:00:00Z").to_pydatetime()

    out = fm._fetch_latest_child_forecasts_in_timerange(
        parent, target_start, target_end
    )

    assert len(out) == 1
    df_out = out[0]
    assert list(df_out.columns) == ["timestamp", "CH1"]
    assert df_out["timestamp"].min() == child_full["timestamp"].min()
    assert df_out["timestamp"].max() == child_full["timestamp"].max()


def test_fetch_latest_child_forecasts_returns_none_if_child_missing(
    monkeypatch, fm, asset_factory
):
    parent = make_parent_with_children(asset_factory, "CH1")

    monkeypatch.setattr(fm, "_get_latest_forecast_df", MagicMock(return_value=None))

    out = fm._fetch_latest_child_forecasts_in_timerange(
        parent,
        now_utc(),
        now_utc(),
    )
    assert out is None


def test_fetch_latest_child_forecasts_returns_none_if_no_common_window(
    monkeypatch, fm, asset_factory
):
    # Parent with two children
    parent = asset_factory(
        asset_uuid="PARENT",
        children=[asset_factory("CH1"), asset_factory("CH2")],
    )

    # CH1 covers 00:00..03:00
    ch1_df = df_child("2025-01-01T00:00:00Z", periods=4)

    # CH2 covers 04:00..07:00 (no overlap with CH1)
    ch2_df = df_child("2025-01-01T04:00:00Z", periods=4)

    def fake_get_latest(asset_id, *_args, **_kwargs):
        return {"CH1": ch1_df, "CH2": ch2_df}[asset_id]

    monkeypatch.setattr(fm, "_get_latest_forecast_df", fake_get_latest)

    out = fm._fetch_latest_child_forecasts_in_timerange(
        parent,
        now_utc(),
        now_utc(),
    )

    assert out is None


def test_fetch_latest_child_forecasts_returns_none_if_child_has_no_points_in_common_window(
    monkeypatch, fm, asset_factory
):
    parent = asset_factory(
        asset_uuid="PARENT",
        children=[asset_factory("CH1"), asset_factory("CH2")],
    )

    ch1_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T03:00:00Z"], utc=True
            ),
            "forecast": [1.0, 2.0],
        }
    )
    ch2_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"], utc=True
            ),
            "forecast": [10.0, 20.0],
        }
    )

    monkeypatch.setattr(
        fm,
        "_get_latest_forecast_df",
        lambda asset_id, *_: {"CH1": ch1_df, "CH2": ch2_df}[asset_id],
    )

    out = fm._fetch_latest_child_forecasts_in_timerange(
        parent,
        now_utc(),
        now_utc(),
    )

    assert out is None


# ============================
# _get_target_horizon
# ============================


def test_get_target_horizon_returns_now_and_now_plus_horizon(fm):
    start, end = fm._get_target_horizon()

    assert isinstance(start, datetime)
    assert isinstance(end, datetime)
    assert start.tzinfo is timezone.utc
    assert end.tzinfo is timezone.utc

    delta_minutes = (end - start).total_seconds() / 60
    assert delta_minutes == DEFAULT_FORECAST_HORIZON_MINUTES


# ============================
# _save_forecast_results
# ============================


def test_save_forecast_results_calls_io_correctly(fm, patch_constraints_io):
    fm.pj_io = MagicMock()
    fm.pj_io.get_or_create.return_value = 123

    fm.forecast_run_io = MagicMock()
    fm.forecast_run_io.create.return_value = "FRID"

    fm.forecast_io = MagicMock()

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "forecast": [1.0, 2.0],
        }
    )

    pj = MagicMock(resolution_minutes=60)

    fm._save_forecast_results(df, pj, "RUNID", "A1")

    fm.pj_io.get_or_create.assert_called_once_with(pj)
    fm.forecast_run_io.create.assert_called_once()
    fm.forecast_io.from_df.assert_called_once_with(df, "FRID")
    patch_constraints_io.from_forecast.assert_called_once_with(df, "FRID", "A1")

    _, kwargs = fm.forecast_run_io.create.call_args
    assert kwargs["start_time"] == df["timestamp"].min().to_pydatetime()
    assert kwargs["end_time"] == df["timestamp"].max().to_pydatetime()
    assert kwargs["frequency_min"] == 60


# ============================
# _run_hierarchy_based_forecast
# ============================


def test_run_hierarchy_based_forecast_happy_path(monkeypatch, fm, asset_factory):
    parent = asset_factory(
        asset_uuid="PARENT",
        children=[asset_factory("CH1")],
    )

    # Deterministic horizon
    t0 = datetime(2026, 1, 10, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(fm, "_get_target_horizon", MagicMock(return_value=(t0, t1)))

    # Child dfs already sliced/renamed as expected by _aggregate_child_forecasts
    child_dfs = [
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
                "CH1": [1.0, 2.0],
            }
        )
    ]
    monkeypatch.setattr(
        fm,
        "_fetch_latest_child_forecasts_in_timerange",
        MagicMock(return_value=child_dfs),
    )

    # Keep aggregation real or stub it; either is fine.
    # Here we keep it real and just assert output.
    result = fm._run_hierarchy_based_forecast(parent)

    assert result is not None
    assert result.model_run_id is None
    assert result.prediction_job.id == "PARENT"
    assert list(result.forecast_df.columns) == ["timestamp", "forecast"]
    assert list(result.forecast_df["forecast"]) == [1.0, 2.0]

    fm._fetch_latest_child_forecasts_in_timerange.assert_called_once_with(
        parent, t0, t1
    )


@pytest.mark.parametrize("child_dfs", [None, [], False])
def test_run_hierarchy_based_forecast_returns_none_if_no_children(
    monkeypatch, fm, asset_factory, child_dfs
):
    parent = asset_factory(
        asset_uuid="PARENT",
        children=[asset_factory("CH1")],
    )

    monkeypatch.setattr(
        fm,
        "_get_target_horizon",
        MagicMock(return_value=(now_utc(), now_utc())),
    )
    monkeypatch.setattr(
        fm,
        "_fetch_latest_child_forecasts_in_timerange",
        MagicMock(return_value=child_dfs),
    )

    out = fm._run_hierarchy_based_forecast(parent)
    assert out is None
