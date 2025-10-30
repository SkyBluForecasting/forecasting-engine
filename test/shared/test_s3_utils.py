import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import patch, MagicMock
from forecasting_engine.shared.s3_utils import (
    find_matching_key,
    load_training_csv_from_s3,
    load_training_pd_from_s3,
    get_s3_client,
    list_training_fsa_ids,
)

# ---------------------------
# Fixtures
# ---------------------------


@pytest.fixture(autouse=True)
def patch_s3_bucket(monkeypatch):
    """Force S3_BUCKET import reference to a dummy value."""
    monkeypatch.setattr("forecasting_engine.shared.s3_utils.S3_BUCKET", "dummy-bucket")


@pytest.fixture
def mock_s3_client():
    client = MagicMock()
    with patch(
        "forecasting_engine.shared.s3_utils.get_s3_client",
        return_value=client,
    ):
        yield client


# ---------------------------
# Tests: get_s3_client
# ---------------------------


def test_get_s3_client_returns_s3_client():
    client = get_s3_client()
    # Basic boto3 client sanity check
    assert client.meta.service_model.service_name == "s3"


# ---------------------------
# Tests: find_matching_key
# ---------------------------


def test_find_matching_key_found(mock_s3_client):
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


def test_find_matching_key_not_found(mock_s3_client):
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": "training/other.csv"}]}]
    mock_s3_client.get_paginator.return_value = paginator

    with pytest.raises(FileNotFoundError):
        find_matching_key("99999", "training/")


def test_find_matching_key_multiple_matches_logs_warning(mock_s3_client):
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

    with patch("forecasting_engine.shared.s3_utils.logger.warning") as mock_warn:
        key = find_matching_key("12345", "training/")
        mock_warn.assert_called_once()
        assert "Multiple files found" in mock_warn.call_args[0][0]
        assert key == "training/12345_train.csv"


# ---------------------------
# Tests: load_training_csv_from_s3
# ---------------------------


def test_load_training_csv_from_s3_valid(mock_s3_client):
    csv_data = "datetime,load,temp\n2024-01-01 00:00,100,21"
    mock_s3_client.get_object.return_value = {"Body": BytesIO(csv_data.encode("utf-8"))}

    df = load_training_csv_from_s3("training/test.csv")
    assert "load" in df.columns
    assert df.index.name == "datetime"
    assert df.iloc[0]["load"] == 100


def test_load_training_csv_from_s3_missing_datetime(mock_s3_client):
    csv_data = "timestamp,load\n2024-01-01 00:00,100"
    mock_s3_client.get_object.return_value = {"Body": BytesIO(csv_data.encode("utf-8"))}

    with pytest.raises(ValueError, match="must contain a 'datetime' column"):
        load_training_csv_from_s3("training/missing_datetime.csv")


def test_load_training_csv_from_s3_missing_load(mock_s3_client):
    csv_data = "datetime,temp\n2024-01-01 00:00,20"
    mock_s3_client.get_object.return_value = {"Body": BytesIO(csv_data.encode("utf-8"))}

    with pytest.raises(ValueError, match="must contain a 'load' column"):
        load_training_csv_from_s3("training/missing_load.csv")


def test_load_training_csv_from_s3_empty_file(mock_s3_client):
    mock_s3_client.get_object.return_value = {"Body": BytesIO(b"")}
    with pytest.raises(
        ValueError, match="CSV at dummy-bucket/training/empty.csv is empty"
    ):
        load_training_csv_from_s3("training/empty.csv")


# ---------------------------
# Tests: load_training_pd_from_s3
# ---------------------------


@patch("forecasting_engine.shared.s3_utils.find_matching_key")
@patch("forecasting_engine.shared.s3_utils.load_training_csv_from_s3")
def test_load_training_pd_from_s3_success(mock_load_csv, mock_find_key):
    mock_find_key.return_value = "training/ABC_train.csv"
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "load": [1, 2],
        }
    ).set_index("datetime")
    mock_load_csv.return_value = df

    result = load_training_pd_from_s3("ABC")
    assert result.equals(df)
    mock_find_key.assert_called_once_with(asset_id="ABC", prefix="training/")
    mock_load_csv.assert_called_once_with("training/ABC_train.csv")


# ---------------------------
# Tests: list_training_fsa_ids
# ---------------------------


def test_list_training_fsa_ids_returns_sorted_list(mock_s3_client):
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "training/B_train.csv"},
                {"Key": "training/A_train.csv"},
                {"Key": "training/Z_train.csv"},
            ]
        }
    ]
    mock_s3_client.get_paginator.return_value = paginator

    result = list_training_fsa_ids()
    assert result == ["A", "B", "Z"]
    mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")


def test_list_training_fsa_ids_filters_non_csv(mock_s3_client):
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

    assert list_training_fsa_ids() == ["K0A"]


def test_list_training_fsa_ids_empty(mock_s3_client):
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": []}]
    mock_s3_client.get_paginator.return_value = paginator

    assert list_training_fsa_ids() == []
