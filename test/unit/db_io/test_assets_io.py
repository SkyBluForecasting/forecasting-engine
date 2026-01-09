import pytest
import pandas as pd

from forecasting_db.models import Asset
from forecasting_engine.db_io.assets_io import AssetsIO

# ----------------------------
# list_assets tests
# ----------------------------


def test_list_assets_returns_all(in_memory_session):
    """Should return all assets if no type filter is given."""
    asset1 = Asset(
        asset_uuid="A1", asset_type="system", name="Asset 1", depth=0, measured=True
    )
    asset2 = Asset(
        asset_uuid="A2",
        asset_type="distribution_substation",
        name="Asset 2",
        depth=0,
        measured=True,
    )
    in_memory_session.add_all([asset1, asset2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    assets = io.list_assets()

    uuids = {a.asset_uuid for a in assets}
    assert uuids == {"A1", "A2"}


def test_list_assets_filters_by_type(in_memory_session):
    """Should filter assets by asset_type."""
    asset1 = Asset(
        asset_uuid="A1", asset_type="system", name="Asset 1", depth=0, measured=True
    )
    asset2 = Asset(
        asset_uuid="A2",
        asset_type="distribution_substation",
        name="Asset 2",
        depth=0,
        measured=True,
    )
    in_memory_session.add_all([asset1, asset2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    assets = io.list_assets(asset_type="system")

    uuids = [a.asset_uuid for a in assets]
    assert uuids == ["A1"]


# ----------------------------
# get_asset tests
# ----------------------------


def test_get_asset_found(in_memory_session):
    """Should return a single asset if found."""
    asset = Asset(
        asset_uuid="A1", asset_type="system", name="Asset 1", depth=0, measured=True
    )
    in_memory_session.add(asset)
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.get_asset("A1")

    assert result.asset_uuid == "A1"


def test_get_asset_not_found(in_memory_session):
    """Should return None if asset not found."""
    io = AssetsIO(in_memory_session)
    result = io.get_asset("NONEXISTENT")

    assert result is None


# ----------------------------
# to_df tests
# ----------------------------


def test_to_df_returns_dataframe(in_memory_session):
    """Should convert assets to a DataFrame."""
    asset = Asset(
        asset_uuid="A1",
        name="Asset 1",
        asset_type="system",
        capacity_kw=100,
        depth=0,
        measured=True,
        parent_uuid=None,
    )
    in_memory_session.add(asset)
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
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


def test_to_df_empty_returns_empty_df(in_memory_session):
    """Should return empty DataFrame when no assets found."""
    io = AssetsIO(in_memory_session)
    df = io.to_df()
    assert df.empty


def test_to_df_filters_by_type(in_memory_session):
    """Should call list_assets with asset_type when provided."""
    asset1 = Asset(
        asset_uuid="A1", asset_type="system", name="Asset 1", depth=0, measured=True
    )
    asset2 = Asset(
        asset_uuid="A2",
        asset_type="distribution_substation",
        name="Asset 2",
        depth=0,
        measured=True,
    )
    in_memory_session.add_all([asset1, asset2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    df = io.to_df(asset_type="distribution_substation")

    uuids = df["asset_uuid"].tolist()
    assert uuids == ["A2"]


# ----------------------------
# from_df tests
# ----------------------------


def test_from_df_not_implemented(in_memory_session):
    """Should raise NotImplementedError."""
    io = AssetsIO(in_memory_session)
    df = pd.DataFrame()
    with pytest.raises(NotImplementedError, match="from_df"):
        io.from_df(df)


# ----------------------------
# list_leaf_assets tests
# ----------------------------


def test_list_leaf_assets_basic(in_memory_session):
    # Hierarchy: PARENT1 -> CHILD1 -> LEAF1
    parent = Asset(
        asset_uuid="PARENT1", asset_type="system", name="Parent", depth=0, measured=True
    )
    child = Asset(
        asset_uuid="CHILD1",
        asset_type="system",
        name="Child",
        depth=1,
        measured=True,
        parent_uuid="PARENT1",
    )
    leaf = Asset(
        asset_uuid="LEAF1",
        asset_type="system",
        name="Leaf",
        depth=2,
        measured=True,
        parent_uuid="CHILD1",
    )

    in_memory_session.add_all([parent, child, leaf])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_leaf_assets()

    # Only LEAF1 has no children → leaf node
    leaf_uuids = [a.asset_uuid for a in result]
    assert leaf_uuids == ["LEAF1"]


def test_list_leaf_assets_top_level_leaf(in_memory_session):
    # Top-level leaf (no parent)
    leaf = Asset(
        asset_uuid="LEAF1", asset_type="system", name="Leaf", depth=0, measured=True
    )
    in_memory_session.add(leaf)
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_leaf_assets()

    assert [a.asset_uuid for a in result] == ["LEAF1"]


def test_list_leaf_assets_multiple_leaves(in_memory_session):
    # Multiple leaves
    parent = Asset(
        asset_uuid="PARENT1", asset_type="system", name="Parent", depth=0, measured=True
    )
    leaf1 = Asset(
        asset_uuid="LEAF1",
        asset_type="system",
        name="Leaf1",
        depth=1,
        measured=True,
        parent_uuid="PARENT1",
    )
    leaf2 = Asset(
        asset_uuid="LEAF2",
        asset_type="distribution_substation",
        name="Leaf2",
        depth=0,
        measured=True,
    )
    in_memory_session.add_all([parent, leaf1, leaf2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_leaf_assets()

    leaf_uuids = {a.asset_uuid for a in result}
    assert leaf_uuids == {"LEAF1", "LEAF2"}


def test_list_leaf_assets_filter_by_type(in_memory_session):
    parent = Asset(
        asset_uuid="PARENT1", asset_type="system", name="Parent", depth=0, measured=True
    )
    leaf1 = Asset(
        asset_uuid="LEAF1",
        asset_type="system",
        name="Leaf1",
        depth=1,
        measured=True,
        parent_uuid="PARENT1",
    )
    leaf2 = Asset(
        asset_uuid="LEAF2",
        asset_type="distribution_substation",
        name="Leaf2",
        depth=0,
        measured=True,
    )
    in_memory_session.add_all([parent, leaf1, leaf2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_leaf_assets(asset_type="system")

    # Only LEAF1 is of type "system"
    leaf_uuids = [a.asset_uuid for a in result]
    assert leaf_uuids == ["LEAF1"]
