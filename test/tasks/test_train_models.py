import pytest
import pandas as pd
from unittest.mock import patch
from forecasting_engine.tasks.Train_models import (
    split_train_test,
    build_pj,
    train_single_fsa,
    train_all,
)


class TestSplitTrainTest:
    def test_basic_split(self):
        df = pd.DataFrame({"a": range(100)})
        train_data = split_train_test(df, horizon_min=60, res_min=10, fsa_id="TEST")
        # test_len = 60 // 10 = 6, so last 6 rows removed
        assert len(train_data) == 94
        assert train_data.iloc[-1]["a"] == 93

    def test_not_enough_rows(self):
        df = pd.DataFrame({"a": range(5)})
        with pytest.raises(ValueError):
            split_train_test(df, horizon_min=60, res_min=10, fsa_id="TEST")


class TestBuildPJ:
    def test_defaults(self):
        pj = build_pj("FSA123")
        assert pj.id == "FSA123"
        assert pj.model == "xgb"
        assert pj.quantiles == [10, 30, 50, 70, 90]
        assert pj.forecast_type == "demand"
        assert pj.lat == 52.0
        assert pj.lon == 5.0
        assert pj.save_train_forecasts is True

    def test_custom_model_id(self):
        pj = build_pj("FSA123", model_id="CUSTOM")
        assert pj.id == "CUSTOM"


class TestTrainSingleFSA:

    @patch("forecasting_engine.tasks.Train_models.train_model_pipeline")
    @patch("forecasting_engine.tasks.Train_models.load_training_pd_from_s3")
    def test_success(self, mock_load, mock_train):
        df = pd.DataFrame({"a": range(200)})
        mock_load.return_value = df
        mock_train.return_value = ("train", "val", "test")

        train_single_fsa("TEST")

        mock_load.assert_called_once_with(asset_id="TEST")
        mock_train.assert_called_once()

    @patch("forecasting_engine.tasks.Train_models.train_model_pipeline")
    @patch(
        "forecasting_engine.tasks.Train_models.load_training_pd_from_s3",
        side_effect=FileNotFoundError,
    )
    def test_file_not_found(self, mock_load, mock_train):
        with pytest.raises(FileNotFoundError):
            train_single_fsa("MISSING")
        mock_load.assert_called_once()

    @patch(
        "forecasting_engine.tasks.Train_models.load_training_pd_from_s3",
        side_effect=ValueError("invalid data"),
    )
    def test_value_error_loading(self, mock_load):
        with pytest.raises(ValueError):
            train_single_fsa("INVALID")

    @patch(
        "forecasting_engine.tasks.Train_models.train_model_pipeline",
        side_effect=LookupError("model missing"),
    )
    @patch("forecasting_engine.tasks.Train_models.load_training_pd_from_s3")
    def test_lookup_error_training(self, mock_load, mock_train):
        df = pd.DataFrame({"a": range(200)})
        mock_load.return_value = df
        with pytest.raises(LookupError):
            train_single_fsa("TEST")

    @patch(
        "forecasting_engine.tasks.Train_models.train_model_pipeline",
        side_effect=Exception("training failed"),
    )
    @patch("forecasting_engine.tasks.Train_models.load_training_pd_from_s3")
    def test_other_exception_training(self, mock_load, mock_train):
        df = pd.DataFrame({"a": range(200)})
        mock_load.return_value = df
        with pytest.raises(Exception):
            train_single_fsa("TEST")

    @patch(
        "forecasting_engine.tasks.Train_models.load_training_pd_from_s3",
        side_effect=Exception("other error"),
    )
    def test_other_exception_loading(self, mock_load):
        with pytest.raises(Exception):
            train_single_fsa("TEST")


class TestTrainAll:

    @patch("forecasting_engine.tasks.Train_models.train_single_fsa")
    @patch("forecasting_engine.tasks.Train_models.list_training_fsa_ids")
    def test_train_all_calls_all_keys(self, mock_list_keys, mock_train_single):
        mock_list_keys.return_value = ["A", "B"]
        mock_train_single.return_value = None

        results = train_all()
        assert results == [None, None]
        mock_train_single.assert_any_call("A")
        mock_train_single.assert_any_call("B")
        assert mock_train_single.call_count == 2
