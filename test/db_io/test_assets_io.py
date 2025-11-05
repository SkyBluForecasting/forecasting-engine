import pytest
import pandas as pd
from unittest.mock import MagicMock

from forecasting_db.models import Asset
from forecasting_engine.db_io.assets_io import AssetsIO


# ----------------------------
# list_assets tests
# ----------------------------


def test_list_assets_returns_all(make_io, mock_session):
    """Should return all assets if no type filter is given."""
    mock_asset1 = MagicMock(spec=Asset)
    mock_asset2 = MagicMock(spec=Asset)
    mock_session.query.return_value.all.return_value = [mock_asset1, mock_asset2]

    io = make_io(AssetsIO)
    assets = io.list_assets()

    assert assets == [mock_asset1, mock_asset2]
    mock_session.query.assert_called_once_with(Asset)


def test_list_assets_filters_by_type(make_io, mock_session):
    """Should filter assets by asset_type."""
    mock_asset = MagicMock(spec=Asset)
    mock_session.query.return_value.filter.return_value.all.return_value = [mock_asset]

    io = make_io(AssetsIO)
    assets = io.list_assets(asset_type="pv")

    mock_session.query.return_value.filter.assert_called_once()
    assert assets == [mock_asset]


# ----------------------------
# get_asset tests
# ----------------------------


def test_get_asset_found(make_io, mock_session):
    """Should return a single asset if found."""
    mock_asset = MagicMock(spec=Asset)
    mock_session.query.return_value.filter.return_value.one_or_none.return_value = (
        mock_asset
    )

    io = make_io(AssetsIO)
    result = io.get_asset("ASSET123")

    assert result == mock_asset
    mock_session.query.return_value.filter.assert_called_once()


def test_get_asset_not_found(make_io, mock_session):
    """Should return None if asset not found."""
    mock_session.query.return_value.filter.return_value.one_or_none.return_value = None

    io = make_io(AssetsIO)
    result = io.get_asset("ASSET123")

    assert result is None


# ----------------------------
# to_df tests
# ----------------------------


def test_to_df_returns_dataframe(make_io, mock_session):
    """Should convert assets to a DataFrame."""
    mock_asset = MagicMock(spec=Asset)
    mock_asset.asset_uuid = "A1"
    mock_asset.name = "Asset 1"
    mock_asset.asset_type = "pv"
    mock_asset.capacity_kw = 100
    mock_asset.parent_uuid = None

    mock_session.query.return_value.all.return_value = [mock_asset]
    io = make_io(AssetsIO)

    df = io.to_df()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert set(df.columns) == {
        "asset_uuid",
        "name",
        "asset_type",
        "capacity_kw",
        "parent_uuid",
    }


def test_to_df_empty_returns_empty_df(make_io, mock_session):
    """Should return empty DataFrame when no assets found."""
    mock_session.query.return_value.all.return_value = []
    io = make_io(AssetsIO)

    df = io.to_df()
    assert df.empty


def test_to_df_filters_by_type(make_io, mock_session):
    """Should call list_assets with asset_type when provided."""
    mock_asset = MagicMock(spec=Asset)
    mock_session.query.return_value.filter.return_value.all.return_value = [mock_asset]

    io = make_io(AssetsIO)
    io.to_df(asset_type="substation")

    mock_session.query.return_value.filter.assert_called_once()


# ----------------------------
# from_df tests
# ----------------------------


def test_from_df_not_implemented(make_io, mock_session):
    """Should raise NotImplementedError."""
    io = make_io(AssetsIO)
    df = pd.DataFrame()
    with pytest.raises(NotImplementedError, match="from_df"):
        io.from_df(df)
