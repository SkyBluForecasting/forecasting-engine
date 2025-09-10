import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import patch, MagicMock
from forecasting_engine.orchestration.s3_utils import (
    find_matching_key,
    load_training_csv_from_s3,
    load_training_pd_from_s3,
    get_s3_bucket,
    get_s3_client,
    extract_asset_id_from_s3_key,
    list_training_fsa_ids,
)

# ---------------------------
# Shared Fixtures
# ---------------------------


@pytest.fixture
def mock_bucket(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "dummy-bucket")


@pytest.fixture(autouse=True)
def patch_get_s3_bucket():
    with patch(
        "forecasting_engine.orchestration.s3_utils.get_s3_bucket",
        return_value="dummy-bucket",
    ):
        yield


@pytest.fixture
def mock_s3_client():
    client = MagicMock()
    with patch(
        "forecasting_engine.orchestration.s3_utils.get_s3_client", return_value=client
    ):
        yield client


# ---------------------------
# Test: Env var
# ---------------------------


def test_get_s3_bucket_success(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    assert get_s3_bucket() == "test-bucket"


def test_get_s3_bucket_env_var_missing(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    with pytest.raises(ValueError, match="S3_BUCKET environment variable is not set"):
        get_s3_bucket()


def test_get_s3_client_returns_s3_client():
    client = get_s3_client()
    # Check it’s a boto3 client instance (simple check)
    assert client.meta.service_model.service_name == "s3"


class TestFindMatchingKey:

    def test_key_found(self, mock_bucket, mock_s3_client):
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "training/12345_train.csv"},
                    {"Key": "training/other.csv"},
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = paginator

        result = find_matching_key("12345", "training/")
        assert result == "training/12345_train.csv"

    def test_key_not_found(self, mock_bucket, mock_s3_client):
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "training/other.csv"}]}
        ]
        mock_s3_client.get_paginator.return_value = paginator

        with pytest.raises(FileNotFoundError):
            find_matching_key("99999", "training/")

    @patch("forecasting_engine.orchestration.s3_utils.get_s3_client")
    def test_multiple_matching_keys_logs_warning(self, mock_get_s3_client, caplog):
        # Setup mock S3 client and paginator
        mock_s3_client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "training/12345_train.csv"},
                    {"Key": "training/12345_backup.csv"},
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = paginator
        mock_get_s3_client.return_value = mock_s3_client

        with patch(
            "forecasting_engine.orchestration.s3_utils.logger.warning"
        ) as mock_warning:
            result = find_matching_key("12345", "training/")

            mock_warning.assert_called_once()
            assert "Multiple files found" in mock_warning.call_args[0][0]

        assert result == "training/12345_train.csv"


class TestLoadCsvFromS3:

    def test_load_valid_csv(self, mock_bucket, mock_s3_client):
        csv_data = (
            "datetime,load,temp\n2024-01-01 00:00,100,21\n2024-01-01 01:00,105,22"
        )
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(csv_data.encode("utf-8"))
        }

        df = load_training_csv_from_s3("training/12345_train.csv")
        assert df.index.name == "datetime"
        assert df.iloc[0]["load"] == 100

    def test_columns_strip_and_order(self, mock_bucket, mock_s3_client):
        csv_data = " datetime , load , temp \n2024-01-01 00:00,100,21"
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(csv_data.encode("utf-8"))
        }

        df = load_training_csv_from_s3("training/strip_test.csv")
        assert list(df.columns) == ["load", "temp"]

    def test_load_training_csv_reorders_load_column(mock_bucket, mock_s3_client):
        csv_data = (
            "datetime,temp,load\n2024-01-01 00:00,21,100\n2024-01-01 01:00,22,105"
        )

        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(csv_data.encode("utf-8"))
        }

        from forecasting_engine.orchestration.s3_utils import load_training_csv_from_s3

        df = load_training_csv_from_s3("training/reorder_test.csv")

        assert df.columns[0] == "load"
        assert "temp" in df.columns
        assert df.iloc[0]["load"] == 100

    def test_load_training_csv_missing_datetime_column(mock_bucket, mock_s3_client):
        csv_data = "timestamp,load\n2024-01-01 00:00,100"
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(csv_data.encode("utf-8"))
        }

        from forecasting_engine.orchestration.s3_utils import load_training_csv_from_s3

        with pytest.raises(ValueError, match="must contain a 'datetime' column"):
            load_training_csv_from_s3("training/missing_datetime.csv")

    def test_load_training_csv_missing_load_column(mock_bucket, mock_s3_client):
        csv_data = "datetime,temp\n2024-01-01 00:00,21"
        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(csv_data.encode("utf-8"))
        }

        from forecasting_engine.orchestration.s3_utils import load_training_csv_from_s3

        with pytest.raises(ValueError, match="must contain a 'load' column"):
            load_training_csv_from_s3("training/missing_load.csv")

    def test_load_empty_csv_raises_value_error(mock_bucket, mock_s3_client):
        from forecasting_engine.orchestration.s3_utils import load_training_csv_from_s3

        mock_s3_client.get_object.return_value = {"Body": BytesIO("".encode("utf-8"))}

        with pytest.raises(
            ValueError, match="CSV at dummy-bucket/training/empty.csv is empty"
        ):
            load_training_csv_from_s3("training/empty.csv")


class TestLoadTrainingPdFromS3:

    @patch("forecasting_engine.orchestration.s3_utils.find_matching_key")
    @patch("forecasting_engine.orchestration.s3_utils.load_training_csv_from_s3")
    def test_load_training_pd_from_s3_success(
        self, mock_load_training_csv, mock_find_key, mock_bucket
    ):
        mock_find_key.return_value = "training/12345_train.csv"

        df_mock = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
                "load": [100, 105],
            }
        ).set_index("datetime")

        mock_load_training_csv.return_value = df_mock

        df = load_training_pd_from_s3("12345")
        assert df.shape[0] == 2


class TestExtractAssetIdFromS3Key:
    @pytest.mark.parametrize(
        "s3_key, expected_asset_id",
        [
            ("path/to/L9M_train.csv", "L9M"),
            ("L9M_train.csv", "L9M"),
            ("folder/L9M.csv", "L9M"),
            ("L9M.csv", "L9M"),
            ("some/dir/ABC_123_test.txt", "ABC"),
            ("justfilename", "justfilename"),
            ("another/path/XYZ_abc.csv", "XYZ"),
            ("file.with.dots_in_name.csv", "file.with.dots"),
        ],
    )
    def test_extract_asset_id(self, s3_key, expected_asset_id):
        result = extract_asset_id_from_s3_key(s3_key)
        assert result == expected_asset_id


class TestListTrainingCSVKeys:

    def test_happy_path_returns_sorted_keys(self, mock_bucket, mock_s3_client):
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "training/K0A_train.csv"},
                    {"Key": "training/ABC_train.csv"},
                    {"Key": "training/XYZ_train.csv"},
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = paginator

        keys = list_training_fsa_ids()
        # Keys should be sorted alphabetically
        assert keys == [
            "ABC",
            "K0A",
            "XYZ",
        ]
        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")

    def test_filters_non_training_files(self, mock_bucket, mock_s3_client):
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "training/K0A_train.csv"},
                    {"Key": "training/README.md"},
                    {"Key": "training/data.txt"},
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = paginator

        keys = list_training_fsa_ids()
        assert keys == ["K0A"]  # only *_train.csv kept

    def test_empty_bucket_returns_empty_list(self, mock_bucket, mock_s3_client):
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        mock_s3_client.get_paginator.return_value = paginator

        keys = list_training_fsa_ids()
        assert keys == []
