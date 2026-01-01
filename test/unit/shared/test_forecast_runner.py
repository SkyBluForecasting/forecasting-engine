import pytest
import pandas as pd
from unittest.mock import MagicMock
from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)
from forecasting_engine.shared.forecast_runner import (
    ForecastManager,
    run_asset_forecast,
    _run_asset_forecast_inner,
)


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def input_df():
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame({"load": [10, 20, 30]}, index=idx)


@pytest.fixture
def mock_forecast_df(input_df):
    return pd.DataFrame({"forecast": [100, 200, 300], "timestamp": input_df.index})


@pytest.fixture
def fm(monkeypatch, input_df, mock_forecast_df):
    """ForecastManager with MeasurementsIO patched and _run_forecast returning mock data."""
    # Patch MeasurementsIO
    meas_mock = MagicMock()
    meas_mock.to_df.return_value = input_df
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.MeasurementsIO",
        lambda session: meas_mock,
    )

    # Patch other IOs to be simple mocks
    for io_name in ["PredictionJobIO", "ForecastRunIO", "ForecastIO", "ConstraintsIO"]:
        monkeypatch.setattr(
            f"forecasting_engine.shared.forecast_runner.{io_name}",
            lambda session: MagicMock(
                **{
                    method: MagicMock()
                    for method in [
                        "get_or_create",
                        "create",
                        "from_df",
                        "from_forecast",
                    ]
                }
            ),
        )

    fm = ForecastManager(MagicMock())
    fm._run_forecast = MagicMock(return_value=(mock_forecast_df, "mlflow_run_id"))
    return fm


# ----------------------------
# ForecastManager tests
# ----------------------------
class TestForecastManager:
    def test_generate_forecast_success(self, fm, mock_forecast_df):
        df = fm.generate_forecast("ASSET123")
        assert isinstance(df, pd.DataFrame)
        assert "forecast" in df.columns

    def test_generate_forecast_no_measurements(self, fm):
        fm.measurements_io.to_df.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No measurements found"):
            fm.generate_forecast("ASSET_EMPTY")

    def test_generate_forecast_pipeline_error(self, fm):
        fm._run_forecast = MagicMock(side_effect=RuntimeError("Pipeline error"))
        with pytest.raises(RuntimeError, match="Pipeline error"):
            fm.generate_forecast("ASSET_FAIL")

    def test_run_forecast_no_model_found(self, monkeypatch, input_df):
        """Test _run_forecast raises LookupError if MLflow model not found."""
        fm = ForecastManager(session=MagicMock())
        pj = PredictionJobDataClass(
            id="ASSET123",
            model="xgb",
            quantiles=[0.1, 0.5, 0.9],
            forecast_type="demand",
            lat=0.0,
            lon=0.0,
            horizon_minutes=60,
            resolution_minutes=15,
            name="ASSET123",
            hyper_params={},
        )

        # Patch MLflowSerializer to simulate no model found
        fake_serializer = MagicMock()
        fake_serializer._find_models.return_value = pd.DataFrame()
        monkeypatch.setattr(
            "forecasting_engine.shared.forecast_runner.MLflowSerializer",
            lambda uri: fake_serializer,
        )

        # Pipeline & normalization mocks
        monkeypatch.setattr(
            "forecasting_engine.shared.forecast_runner.create_forecast_pipeline_core",
            lambda *a, **k: pd.DataFrame(),
        )
        monkeypatch.setattr(
            "forecasting_engine.shared.forecast_runner.normalize_forecast_columns",
            lambda df: df,
        )

        with pytest.raises(LookupError, match="No model found in MLflow"):
            fm._run_forecast(pj, input_df)


# ----------------------------
# run_asset_forecast wrapper tests
# ----------------------------
def test_run_asset_forecast_success(monkeypatch):
    mock_fm = MagicMock()
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.ForecastManager", mock_fm
    )
    mock_instance = mock_fm.return_value
    result = run_asset_forecast("ASSET123")
    mock_fm.assert_called_once()
    mock_instance.generate_forecast.assert_called_once_with("ASSET123")
    assert result["status"] == "success"


@pytest.mark.parametrize(
    "exception,expected_status",
    [
        (ValueError("bad input"), "bad_input"),
        (FileNotFoundError("file missing"), "not_found"),
        (RuntimeError("fatal"), "fatal_error"),
    ],
)
def test_run_asset_forecast_exceptions(monkeypatch, exception, expected_status):
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner._run_asset_forecast_inner",
        lambda asset_id: (_ for _ in ()).throw(exception),
    )
    result = run_asset_forecast("ASSET123")
    assert result["status"] == expected_status
    assert str(exception) in result["message"]


def test__run_asset_forecast_inner_calls_forecastmanager(monkeypatch):
    """Ensure ForecastManager is called inside context manager."""
    mock_session_instance = MagicMock()
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session_instance
    mock_session.__exit__.return_value = None
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.SessionLocal",
        lambda: mock_session,
    )

    mock_fm_class = MagicMock()
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.ForecastManager",
        mock_fm_class,
    )

    result = _run_asset_forecast_inner("ASSET123")
    mock_fm_class.assert_called_once_with(mock_session_instance)
    mock_fm_class.return_value.generate_forecast.assert_called_once_with("ASSET123")
    mock_session.__enter__.assert_called_once()
    mock_session.__exit__.assert_called_once()
    assert result["status"] == "success"


# ----------------------------
# _run_forecast tests
# ----------------------------
def make_prediction_job():
    return PredictionJobDataClass(
        id="ASSET123",
        model="xgb",
        quantiles=[0.1, 0.5, 0.9],
        forecast_type="demand",
        lat=0.0,
        lon=0.0,
        horizon_minutes=60,
        resolution_minutes=15,
        name="ASSET123",
        hyper_params={},
    )


def test__run_forecast_success(monkeypatch, input_df, mock_forecast_df):
    fm = ForecastManager(session=MagicMock())
    pj = make_prediction_job()

    fake_serializer = MagicMock()
    fake_serializer._find_models.return_value = pd.DataFrame([{"run_id": "RUN123"}])
    fake_serializer.load_model.return_value = ("model_obj", {"spec": "dummy"})
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.MLflowSerializer",
        lambda uri: fake_serializer,
    )

    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.create_forecast_pipeline_core",
        lambda pj, df, model, specs: mock_forecast_df,
    )
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.normalize_forecast_columns",
        lambda df: df,
    )

    forecast_df, run_id = fm._run_forecast(pj, input_df)
    assert forecast_df.equals(mock_forecast_df)
    assert run_id == "RUN123"
    fake_serializer._find_models.assert_called_once()
    fake_serializer.load_model.assert_called_once()


def test__run_forecast_no_model_found(monkeypatch, input_df):
    fm = ForecastManager(session=MagicMock())
    pj = make_prediction_job()

    fake_serializer = MagicMock()
    fake_serializer._find_models.return_value = pd.DataFrame()  # triggers LookupError
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.MLflowSerializer",
        lambda uri: fake_serializer,
    )
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.create_forecast_pipeline_core",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "forecasting_engine.shared.forecast_runner.normalize_forecast_columns",
        lambda df: df,
    )

    import pytest

    with pytest.raises(LookupError, match="No model found in MLflow"):
        fm._run_forecast(pj, input_df)
