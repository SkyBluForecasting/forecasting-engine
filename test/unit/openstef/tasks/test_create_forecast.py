# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from test.unit.openstef.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
import sys

import forecasting_engine.openstef as new_openstef
import forecasting_engine.openstef.tasks.create_forecast as task
from forecasting_engine.openstef.enums import PipelineType
from forecasting_engine.openstef.exceptions import InputDataOngoingFlatlinerError
from forecasting_engine.openstef.model.serializer import MLflowSerializer
from forecasting_engine.openstef.tasks.create_forecast import create_forecast_task

FORECAST_MOCK = "forecast_mock"


class TestCreateForecastTask(TestCase):
    @patch("forecasting_engine.openstef.model.serializer.mlflow.sklearn.load_model")
    @patch(
        "forecasting_engine.openstef.model.serializer.MLflowSerializer._get_model_uri"
    )
    def setUp(self, _get_model_uri_mock, mock_load_model):
        # Shim for old import path
        sys.modules["openstef"] = new_openstef

        # Use test data
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)

        # MLflow serializer
        self.serializer = MLflowSerializer(
            mlflow_tracking_uri="./test/unit/openstef/trained_models/mlruns"
        )

        # Mock model URI
        rel_path = "test/unit/openstef/trained_models/mlruns/893156335105023143/2ca1d126e8724852b303b256e64a6c4f/artifacts/model"
        _get_model_uri_mock.return_value = Path(rel_path).absolute().as_uri()

        # Mock MLflow model load so it does not deserialize real model
        mock_load_model.return_value = MagicMock()

        # Load model (returns mocked object)
        self.model, _ = self.serializer.load_model(experiment_name="307")

    def test_mocked_model_path(self):
        """This test explicitely tests if the model path is mocked correctly"""
        assert (
            "test/unit/openstef/trained_models/mlruns/893156335105023143/2ca1d126e8724852b303b256e64a6c4f/artifacts/model"
            in self.model.path
        )

    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_forecast_task_happy_flow_1(self):
        """Test happy flow of create forecast task."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(context.mock_calls[3][0], "database.write_forecast")
        self.assertEqual(context.mock_calls[3].args[0], FORECAST_MOCK)

    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(return_value=FORECAST_MOCK),
    )
    def test_create_forecast_task_happy_flow(self):
        """Test happy flow of create forecast task."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = [999]

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertNotEqual(
            self.pj.id, context.config.externally_posted_forecasts_pids[0]
        )
        self.assertEqual(context.mock_calls[3][0], "database.write_forecast")
        self.assertEqual(context.mock_calls[3].args[0], FORECAST_MOCK)

    def test_create_forecast_task_skip_external(self):
        """Test that making a forecast is skipped for externally posted pids."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = [307]

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(self.pj.id, context.config.externally_posted_forecasts_pids[0])
        self.assertEqual(
            context.mock_calls[0].args[0],
            "Skip this PredictionJob because its forecasts are posted by an external process.",
        )

    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=InputDataOngoingFlatlinerError()),
    )
    def test_create_forecast_known_zero_flatliner(self):
        """Test that making a forecast is skipped for known zero flatliners."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = [307]

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(self.pj.id, context.config.known_zero_flatliners[0])
        self.assertEqual(
            context.mock_calls[3].args[0],
            "No forecasts were made for this known zero flatliner prediction job. No forecasts need to be made either, since the fallback forecasts are sufficient.",
        )
        assert (
            not context.database.write_forecast.called
        ), "The `write_forecast` method should not have been called."

    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=LookupError()),
    )
    def test_create_forecast_known_zero_flatliner_no_model(self):
        """Test that making a forecast is skipped for known zero flatliners for which no model has been trained yet."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = [307]

        # Act
        create_forecast_task(self.pj, context)

        # Assert
        self.assertEqual(self.pj.id, context.config.known_zero_flatliners[0])
        self.assertEqual(
            context.mock_calls[3].args[0],
            "No forecasts were made for this known zero flatliner prediction job. No forecasts need to be made either, since the fallback forecasts are sufficient.",
        )
        assert (
            not context.database.write_forecast.called
        ), "The `write_forecast` method should not have been called."

    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=InputDataOngoingFlatlinerError()),
    )
    def test_create_forecast_unexpected_zero_flatliner(self):
        """Test that there is an informative error message for unexpected zero flatliners."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = None

        # Act & Assert
        with pytest.raises(InputDataOngoingFlatlinerError) as e:
            create_forecast_task(self.pj, context)

        assert (
            e.value.args[0]
            == 'All recent load measurements are constant. Check the load profile of this pid as well as related/neighbouring prediction jobs. Afterwards, consider adding this pid to the "known_zero_flatliners" app_setting and possibly removing other pids from the same app_setting.'
        )

    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=LookupError("Model not found. First train a model!")),
    )
    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.detect_ongoing_flatliner",
        MagicMock(return_value=True),
    )
    def test_create_forecast_unexpected_zero_flatliner_lookuperror(self):
        """Test that the lookuperror is properly raised when the prediction job is an unexpected zero flatliner."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = None

        # Act & Assert
        with pytest.raises(LookupError) as e:
            create_forecast_task(self.pj, context)

        assert (
            e.value.args[0]
            == 'Model not found. Consider checking for a flatliner and adding this pid to the "known_zero_flatliners" app_setting. For flatliners, no model can be trained.'
        )

    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline",
        MagicMock(side_effect=LookupError("Model not found. First train a model!")),
    )
    @patch(
        "forecasting_engine.openstef.tasks.create_forecast.detect_ongoing_flatliner",
        MagicMock(return_value=False),
    )
    def test_create_forecast_lookuperror(self):
        """Test that the lookuperror is properly raised when the prediction job is not a zero flatliner."""
        # Arrange
        context = MagicMock()
        context.config.externally_posted_forecasts_pids = None
        context.config.known_zero_flatliners = None

        # Act & Assert
        with pytest.raises(LookupError) as e:
            create_forecast_task(self.pj, context)

        assert e.value.args[0] == "Model not found. First train a model!"

    @patch("forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline")
    def test_create_forecast_task_train_only(self, create_forecast_pipeline_mock):
        """Test happy flow of create forecast task for train only pj."""
        context = MagicMock()
        pj = self.pj
        pj.pipelines_to_run = [PipelineType.TRAIN]
        create_forecast_task(pj, context)
        self.assertEqual(create_forecast_pipeline_mock.call_count, 0)

    @patch("forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline")
    def test_create_forecast_task_forecast_only(self, create_forecast_pipeline_mock):
        """Test happy flow of create forecast task for forecast only pj."""
        # Arrange
        context = MagicMock()
        create_forecast_pipeline_mock.return_value = FORECAST_MOCK
        pj = self.pj
        pj.pipelines_to_run = [PipelineType.FORECAST]

        # Act
        create_forecast_task(pj, context)

        # Assert
        self.assertEqual(create_forecast_pipeline_mock.call_count, 1)
        self.assertEqual(context.mock_calls[5].args[0], FORECAST_MOCK)

    @patch("forecasting_engine.openstef.tasks.create_forecast.create_forecast_pipeline")
    @patch("forecasting_engine.openstef.model.serializer.MLflowSerializer.load_model")
    @patch("forecasting_engine.openstef.tasks.utils.taskcontext.post_teams")
    def test_create_forecast_task_with_context(
        self, post_teams_mock, load_model_mock, create_forecast_pipeline_mock
    ):
        """Test create_forecast_task with context, fully mocked to avoid filesystem and MLflow."""

        # --- Arrange ---
        context_mock = MagicMock()  # noqa
        dbmock = MagicMock()

        # Use prediction job and modelspecs from setUp
        dbmock.get_prediction_jobs.return_value = [self.pj, self.pj]
        dbmock.get_modelspecs.return_value = self.modelspecs

        # Mock model load (returns the model from setUp)
        load_model_mock.return_value = self.model

        # Mock forecast pipeline to return predictable forecast
        create_forecast_pipeline_mock.return_value = FORECAST_MOCK

        # Patch TaskContext to use our context mock
        configmock_taskcontext = MagicMock()
        configmock_taskcontext.return_value.paths_mlflow_tracking_uri = "./ignored"
        configmock_taskcontext.return_value.paths_artifact_folder = "./ignored"

        # --- Act ---
        task.main(config=configmock_taskcontext(), database=dbmock)

        # --- Assert ---
        # Check that write_forecast was called with our mock forecast
        written_forecast = dbmock.write_forecast.call_args.args[0]
        self.assertEqual(written_forecast, FORECAST_MOCK)

        # Check that the pipeline was called at least once
        self.assertGreaterEqual(create_forecast_pipeline_mock.call_count, 1)
