import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from forecasting_engine.orchestration.forecast_runner import (
    build_prediction_job,
    run_openstef_forecast,
    generate_forecast_for_asset,
)


# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def pj():
    return build_prediction_job("ASSET1")


@pytest.fixture
def input_df():
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame({"load": [10, 20, 30]}, index=idx)


@pytest.fixture
def mock_forecast_df(input_df):
    return pd.DataFrame({"forecast": [100, 200, 300], "timestamp": input_df.index})


# ----------------------------
# Tests for build_prediction_job
# ----------------------------


class TestBuildPredictionJob:

    def test_build_prediction_job_fields(self):
        pj = build_prediction_job("ASSET123")
        assert pj.id == "ASSET123"
        assert pj.model == "xgb"
        assert pj.horizon_minutes == 48 * 60
        assert pj.resolution_minutes == 60
        assert pj.forecast_type == "demand"
        assert isinstance(pj.quantiles, list)
        assert pj.save_train_forecasts is True


# ----------------------------
# Tests for run_openstef_forecast
# ----------------------------


class TestRunOpenstefForecast:

    def test_run_openstef_forecast_success(self, pj, input_df, mock_forecast_df):
        with patch(
            "forecasting_engine.orchestration.forecast_runner.MLflowSerializer"
        ) as mock_serializer_cls, patch(
            "forecasting_engine.orchestration.forecast_runner.create_forecast_pipeline_core"
        ) as mock_pipeline:

            mock_serializer = MagicMock()
            mock_serializer_cls.return_value = mock_serializer
            mock_serializer._find_models.return_value = pd.DataFrame(
                [{"run_id": "123"}]
            )
            mock_serializer.load_model.return_value = ("model", "specs")

            mock_pipeline.return_value = mock_forecast_df

            df, run_id = run_openstef_forecast(pj, input_df, "mlflow_uri")
            pd.testing.assert_frame_equal(df, mock_forecast_df)
            assert run_id == "123"

    def test_run_openstef_forecast_no_model(self, pj, input_df):
        with patch(
            "forecasting_engine.orchestration.forecast_runner.MLflowSerializer"
        ) as mock_serializer_cls:
            mock_serializer = MagicMock()
            mock_serializer_cls.return_value = mock_serializer
            mock_serializer._find_models.return_value = pd.DataFrame()  # no model

            with pytest.raises(LookupError, match="No model found"):
                run_openstef_forecast(pj, input_df, "mlflow_uri")

    def test_run_openstef_forecast_pipeline_error(self, pj, input_df):
        with patch(
            "forecasting_engine.orchestration.forecast_runner.MLflowSerializer"
        ) as mock_serializer_cls, patch(
            "forecasting_engine.orchestration.forecast_runner.create_forecast_pipeline_core"
        ) as mock_pipeline:

            mock_serializer = MagicMock()
            mock_serializer_cls.return_value = mock_serializer
            mock_serializer._find_models.return_value = pd.DataFrame(
                [{"run_id": "123"}]
            )
            mock_serializer.load_model.return_value = ("model", "specs")

            mock_pipeline.side_effect = RuntimeError("Pipeline failed")

            with pytest.raises(RuntimeError, match="Pipeline failed"):
                run_openstef_forecast(pj, input_df, "mlflow_uri")


# ----------------------------
# Tests for ForecastManager.generate_forecast
# ----------------------------


class TestForecastManager:

    @pytest.fixture(autouse=True)
    def patch_session_io(self, mock_forecast_df, input_df):
        with patch(
            "forecasting_engine.orchestration.forecast_runner.SessionLocal"
        ) as mock_session_cls, patch(
            "forecasting_engine.orchestration.forecast_runner.MeasurementsIO"
        ) as mock_meas_io_cls, patch(
            "forecasting_engine.orchestration.forecast_runner.PredictionJobIO"
        ) as mock_pj_io_cls, patch(
            "forecasting_engine.orchestration.forecast_runner.ForecastRunIO"
        ) as mock_fr_io_cls, patch(
            "forecasting_engine.orchestration.forecast_runner.ForecastIO"
        ) as mock_f_io_cls, patch(
            "forecasting_engine.orchestration.forecast_runner.ConstraintsIO"
        ) as mock_constraints_io_cls, patch(
            "forecasting_engine.orchestration.forecast_runner.run_openstef_forecast"
        ) as mock_run_forecast, patch(
            "forecasting_engine.orchestration.forecast_runner.ForecastDataProcessor"
        ) as mock_processor_cls:

            # Mock session
            mock_session = MagicMock()
            mock_session_cls.return_value.__enter__.return_value = mock_session

            # Mock measurements
            mock_meas_io = MagicMock()
            mock_meas_io.to_df.return_value = input_df
            mock_meas_io_cls.return_value = mock_meas_io

            # Mock other IO classes
            mock_pj_io_cls.return_value.get_or_create.return_value = "pj_id"
            mock_fr_io_cls.return_value.create.return_value = "fr_id"
            mock_f_io_cls.return_value.from_df.return_value = None
            mock_constraints_io_cls.return_value.from_forecast.return_value = None

            # Mock forecast
            mock_run_forecast.return_value = (mock_forecast_df, "mlflow_run_id")

            # Mock processor
            mock_processor = MagicMock()
            mock_processor.add_forecast_horizon_nans.return_value = input_df
            mock_processor_cls.return_value = mock_processor

            self.mocks = {
                "session": mock_session,
                "measurements_io": mock_meas_io,
                "pj_io": mock_pj_io_cls.return_value,
                "forecast_run_io": mock_fr_io_cls.return_value,
                "forecast_io": mock_f_io_cls.return_value,
                "constraints_io": mock_constraints_io_cls.return_value,
                "run_forecast": mock_run_forecast,
                "processor": mock_processor,
            }

            yield

    def test_generate_forecast_success(self):
        df = generate_forecast_for_asset("ASSET123")
        assert isinstance(df, pd.DataFrame)
        assert "forecast" in df.columns

        # Confirm all main steps were called
        self.mocks["measurements_io"].to_df.assert_called_once_with("ASSET123")
        self.mocks["processor"].add_forecast_horizon_nans.assert_called_once()
        self.mocks["pj_io"].get_or_create.assert_called_once()
        self.mocks["forecast_run_io"].create.assert_called_once()
        self.mocks["forecast_io"].from_df.assert_called_once()
        self.mocks["constraints_io"].from_forecast.assert_called_once()
        self.mocks["run_forecast"].assert_called_once()

    def test_generate_forecast_no_measurements(self):
        self.mocks["measurements_io"].to_df.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No measurements found"):
            generate_forecast_for_asset("ASSET_EMPTY")

    def test_generate_forecast_pipeline_error(self):
        self.mocks["run_forecast"].side_effect = RuntimeError("Pipeline error")
        with pytest.raises(RuntimeError, match="Pipeline error"):
            generate_forecast_for_asset("ASSET_FAIL")
