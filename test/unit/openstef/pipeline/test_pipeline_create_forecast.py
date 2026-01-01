# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime as dt
from pathlib import Path
import sys
import pandas as pd
from test.unit.openstef.utils.base import BaseTestCase
from test.unit.openstef.utils.data import TestData
from unittest.mock import MagicMock, patch

from mlflow.exceptions import MlflowException
import forecasting_engine.openstef as new_openstef

from forecasting_engine.openstef.model.serializer import MLflowSerializer
from forecasting_engine.openstef.pipeline import create_forecast, utils


# Absolute project root (robust on CI)
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TestCreateForecastPipeline(BaseTestCase):
    @patch("forecasting_engine.openstef.model.serializer.mlflow.sklearn.load_model")
    @patch(
        "forecasting_engine.openstef.model.serializer.MLflowSerializer._get_model_uri"
    )
    def setUp(self, _get_model_uri_mock, mock_load_model) -> None:
        super().setUp()

        # Shim for legacy `import openstef`
        sys.modules["openstef"] = new_openstef

        self.pj = TestData.get_prediction_job(pid=307)

        self.serializer = MLflowSerializer(
            mlflow_tracking_uri=str(
                PROJECT_ROOT / "test/unit/openstef/trained_models/mlruns"
            )
        )

        # Load test data
        self.data = TestData.load("reference_sets/307-test-data.csv")
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

        # Mock model URI
        rel_path = (
            PROJECT_ROOT
            / "test/unit/openstef/trained_models/mlruns"
            / "893156335105023143"
            / "2ca1d126e8724852b303b256e64a6c4f"
            / "artifacts/model"
        )
        _get_model_uri_mock.return_value = rel_path.as_uri()

        # Mock MLflow model load (NO confidence_interval here)
        mock_load_model.return_value = MagicMock()

        # Load model
        self.model, self.model_specs = self.serializer.load_model(experiment_name="307")

        # 🔒 Force correct stdev type
        self.model.standard_deviation = pd.Series(
            [0.1],
            index=[pd.Timedelta(minutes=15)],
        )

        # 🔒 Kill legacy float path
        self.model.confidence_interval = None

    def test_generate_forecast_datetime_range_single_null_values_target_column(self):
        time_format = "%Y-%m-%d %H:%M:%S%z"
        forecast_start_expected = dt.strptime("2020-11-26 00:00:00+0000", time_format)
        forecast_end_expected = dt.strptime("2020-11-30 00:00:00+0000", time_format)

        forecast_data = self.data.copy()
        forecast_data.loc["2020-11-26":"2020-12-01", forecast_data.columns[0]] = None

        forecast_start, forecast_end = utils.generate_forecast_datetime_range(
            forecast_data=forecast_data
        )

        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    def test_generate_forecast_datetime_range_multiple_null_values_target_column(self):
        time_format = "%Y-%m-%d %H:%M:%S%z"
        forecast_start_expected = dt.strptime("2020-11-26 00:00:00+0000", time_format)
        forecast_end_expected = dt.strptime("2020-11-30 00:00:00+0000", time_format)

        forecast_data = self.data.copy()
        forecast_data.loc["2020-11-26":"2020-12-01", forecast_data.columns[0]] = None
        forecast_data.loc["2020-11-23":"2020-11-24", forecast_data.columns[0]] = None

        forecast_start, forecast_end = utils.generate_forecast_datetime_range(
            forecast_data=forecast_data
        )

        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    def test_generate_forecast_datetime_range_not_null_values_target_column(self):
        forecast_data = self.data.copy()
        forecast_data.loc["2020-11-26":"2020-12-01", forecast_data.columns[0]] = 1

        with self.assertRaises(ValueError):
            utils.generate_forecast_datetime_range(forecast_data)

    def test_generate_forecast_datetime_range_only_null_values_target_column(self):
        time_format = "%Y-%m-%d %H:%M:%S%z"
        forecast_start_expected = dt.strptime("2020-10-31 00:45:00+0000", time_format)
        forecast_end_expected = dt.strptime("2020-11-30 00:00:00+0000", time_format)

        forecast_data = self.data.copy()
        forecast_data.loc[:, forecast_data.columns[0]] = None

        forecast_start, forecast_end = utils.generate_forecast_datetime_range(
            forecast_data=forecast_data
        )

        self.assertEqual(forecast_start, forecast_start_expected)
        self.assertEqual(forecast_end, forecast_end_expected)

    @patch("forecasting_engine.openstef.validation.validation.is_data_sufficient")
    @patch("forecasting_engine.openstef.model.serializer.MLflowSerializer.load_model")
    def test_create_forecast_pipeline_incomplete_inputdata(
        self, load_model_mock, is_data_sufficient_mock
    ):
        is_data_sufficient_mock.return_value = False
        load_model_mock.return_value = (self.model, self.model_specs)

        forecast_data = self.data.copy()
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28":"2020-12-01", col_name] = None

        model, model_specs = self.serializer.load_model("307")

        # ✅ Mock standard_deviation with required columns
        model.standard_deviation = pd.DataFrame(
            {"hour": [0, 1, 2], "horizon": [0, 0, 0], "stdev": [1.0, 1.0, 1.0]}
        )

        # Add 'hour' column to forecast_data for fallback grouping
        forecast_data["hour"] = forecast_data.index.hour

        # create standard_deviation for all hours present in forecast_data
        hours = forecast_data["hour"].unique()
        model.standard_deviation = pd.DataFrame(
            {"hour": hours, "horizon": [0] * len(hours), "stdev": [1.0] * len(hours)}
        )
        # set 'hour' as index because the code uses stdev.loc[x.hour]
        model.standard_deviation.set_index("hour", inplace=True)

        forecast = create_forecast.create_forecast_pipeline_core(
            pj=self.pj,
            input_data=forecast_data,
            model=model,
            model_specs=model_specs,
        )

        assert "substituted" in forecast.quality.values

    @patch(
        "forecasting_engine.openstef.pipeline.create_forecast.create_forecast_pipeline_core"
    )
    @patch("forecasting_engine.openstef.model.serializer.MLflowSerializer.load_model")
    def test_create_forecast_pipeline_wrong_forecast_pid(self, load_mock, core_mock):
        def side_effect(experiment_name):
            if experiment_name == "307":
                return self.model, self.model_specs
            raise MlflowException("Wrong pid")

        load_mock.side_effect = side_effect
        core_mock.return_value = MagicMock()

        self.pj.alternative_forecast_model_pid = "703"

        forecast_data = self.data.copy()
        forecast_data.loc["2020-11-28":"2020-12-01", forecast_data.columns[0]] = None

        with self.assertRaises(MlflowException):
            create_forecast.create_forecast_pipeline(
                self.pj,
                forecast_data,
                str(PROJECT_ROOT / "test/unit/openstef/trained_models/mlruns"),
            )
