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


# ----------------------------
# list_non_leaf_assets tests
# ----------------------------


def test_list_non_leaf_assets_basic(in_memory_session):
    """
    Hierarchy:
        PARENT (depth 0)
          -> CHILD (depth 1)
              -> LEAF (depth 2)

    Non-leaf assets should be: PARENT, CHILD
    """
    parent = Asset(
        asset_uuid="PARENT",
        asset_type="system",
        name="Parent",
        depth=0,
        measured=True,
    )
    child = Asset(
        asset_uuid="CHILD",
        asset_type="system",
        name="Child",
        depth=1,
        measured=True,
        parent_uuid="PARENT",
    )
    leaf = Asset(
        asset_uuid="LEAF",
        asset_type="system",
        name="Leaf",
        depth=2,
        measured=True,
        parent_uuid="CHILD",
    )

    in_memory_session.add_all([parent, child, leaf])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_non_leaf_assets()

    non_leaf_uuids = {a.asset_uuid for a in result}
    assert non_leaf_uuids == {"PARENT", "CHILD"}


def test_list_non_leaf_assets_no_children_returns_empty(in_memory_session):
    """If there are no parent-child links, there are no non-leaf assets."""
    a1 = Asset(asset_uuid="A1", asset_type="system", name="A1", depth=0, measured=True)
    a2 = Asset(asset_uuid="A2", asset_type="system", name="A2", depth=0, measured=True)
    in_memory_session.add_all([a1, a2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_non_leaf_assets()

    assert result == []


def test_list_non_leaf_assets_filter_by_type(in_memory_session):
    """
    If filtering by type, return only non-leaf assets of that type (not their children).
    """
    parent_system = Asset(
        asset_uuid="P_SYSTEM",
        asset_type="system",
        name="Parent System",
        depth=0,
        measured=True,
    )
    child_system = Asset(
        asset_uuid="C_SYSTEM",
        asset_type="system",
        name="Child System",
        depth=1,
        measured=True,
        parent_uuid="P_SYSTEM",
    )

    parent_sub = Asset(
        asset_uuid="P_SUB",
        asset_type="distribution_substation",
        name="Parent Sub",
        depth=0,
        measured=True,
    )
    child_sub = Asset(
        asset_uuid="C_SUB",
        asset_type="distribution_substation",
        name="Child Sub",
        depth=1,
        measured=True,
        parent_uuid="P_SUB",
    )

    in_memory_session.add_all([parent_system, child_system, parent_sub, child_sub])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_non_leaf_assets(asset_type="system")

    # Only P_SYSTEM is a non-leaf of type system
    assert [a.asset_uuid for a in result] == ["P_SYSTEM"]


# ----------------------------
# list_non_leaf_assets_at_depth tests
# ----------------------------


def test_list_non_leaf_assets_at_depth_returns_only_matching_depth(in_memory_session):
    """
    Build two non-leaf assets at different depths:

    P0 (depth 0) -> C1 (depth 1) -> L2 (depth 2)
    """
    p0 = Asset(asset_uuid="P0", asset_type="system", name="P0", depth=0, measured=True)
    c1 = Asset(
        asset_uuid="C1",
        asset_type="system",
        name="C1",
        depth=1,
        measured=True,
        parent_uuid="P0",
    )
    l2 = Asset(
        asset_uuid="L2",
        asset_type="system",
        name="L2",
        depth=2,
        measured=True,
        parent_uuid="C1",
    )

    in_memory_session.add_all([p0, c1, l2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)

    # Depth 0 non-leaf should be P0
    result0 = io.list_non_leaf_assets_at_depth(0)
    assert [a.asset_uuid for a in result0] == ["P0"]

    # Depth 1 non-leaf should be C1
    result1 = io.list_non_leaf_assets_at_depth(1)
    assert [a.asset_uuid for a in result1] == ["C1"]

    # Depth 2 is leaf; should be empty
    result2 = io.list_non_leaf_assets_at_depth(2)
    assert result2 == []


def test_list_non_leaf_assets_at_depth_orders_by_uuid(in_memory_session):
    """
    Ensure ordering is deterministic: order_by Asset.asset_uuid.asc().
    """
    # P0 has two children at depth 1, both non-leaf (each has its own child)
    p0 = Asset(asset_uuid="P0", asset_type="system", name="P0", depth=0, measured=True)

    # These two are at depth=1 and each has a child -> both are non-leaf
    b1 = Asset(
        asset_uuid="B1",
        asset_type="system",
        name="B1",
        depth=1,
        measured=True,
        parent_uuid="P0",
    )
    a1 = Asset(
        asset_uuid="A1",
        asset_type="system",
        name="A1",
        depth=1,
        measured=True,
        parent_uuid="P0",
    )

    b2 = Asset(
        asset_uuid="B2",
        asset_type="system",
        name="B2",
        depth=2,
        measured=True,
        parent_uuid="B1",
    )
    a2 = Asset(
        asset_uuid="A2",
        asset_type="system",
        name="A2",
        depth=2,
        measured=True,
        parent_uuid="A1",
    )

    in_memory_session.add_all([p0, b1, a1, b2, a2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_non_leaf_assets_at_depth(1)

    # Ordered by asset_uuid asc: A1 then B1
    assert [a.asset_uuid for a in result] == ["A1", "B1"]


def test_list_non_leaf_assets_at_depth_filter_by_type(in_memory_session):
    """
    Same depth, but different types; filter should return only matching type.
    """
    p0 = Asset(asset_uuid="P0", asset_type="system", name="P0", depth=0, measured=True)

    sys_parent = Asset(
        asset_uuid="SYS1",
        asset_type="system",
        name="SYS1",
        depth=1,
        measured=True,
        parent_uuid="P0",
    )
    sys_child = Asset(
        asset_uuid="SYS2",
        asset_type="system",
        name="SYS2",
        depth=2,
        measured=True,
        parent_uuid="SYS1",
    )

    sub_parent = Asset(
        asset_uuid="SUB1",
        asset_type="distribution_substation",
        name="SUB1",
        depth=1,
        measured=True,
        parent_uuid="P0",
    )
    sub_child = Asset(
        asset_uuid="SUB2",
        asset_type="distribution_substation",
        name="SUB2",
        depth=2,
        measured=True,
        parent_uuid="SUB1",
    )

    in_memory_session.add_all([p0, sys_parent, sys_child, sub_parent, sub_child])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    result = io.list_non_leaf_assets_at_depth(1, asset_type="distribution_substation")

    assert [a.asset_uuid for a in result] == ["SUB1"]


# ----------------------------
# get_max_depth_non_leaf tests
# ----------------------------


def test_get_max_depth_non_leaf_basic(in_memory_session):
    """
    P0 (depth 0) -> C1 (depth 1) -> L2 (depth 2)

    Non-leaf assets are P0 (0) and C1 (1). Max non-leaf depth = 1.
    """
    p0 = Asset(asset_uuid="P0", asset_type="system", name="P0", depth=0, measured=True)
    c1 = Asset(
        asset_uuid="C1",
        asset_type="system",
        name="C1",
        depth=1,
        measured=True,
        parent_uuid="P0",
    )
    l2 = Asset(
        asset_uuid="L2",
        asset_type="system",
        name="L2",
        depth=2,
        measured=True,
        parent_uuid="C1",
    )

    in_memory_session.add_all([p0, c1, l2])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    assert io.get_max_depth_non_leaf() == 1


def test_get_max_depth_non_leaf_empty_returns_zero(in_memory_session):
    """No parent-child edges => no non-leaf => should return 0 per implementation."""
    leaf = Asset(
        asset_uuid="LEAF", asset_type="system", name="Leaf", depth=5, measured=True
    )
    in_memory_session.add(leaf)
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)
    assert io.get_max_depth_non_leaf() == 0


def test_get_max_depth_non_leaf_filter_by_type(in_memory_session):
    """
    Two hierarchies of different types:

    system: P0 (0) -> C1 (1) -> L2 (2) => max non-leaf depth = 1
    substation: PS (0) -> CS (1) -> GS (2) -> LS (3) => non-leaf depths 0,1,2 => max = 2
    """
    # system chain
    p0 = Asset(asset_uuid="P0", asset_type="system", name="P0", depth=0, measured=True)
    c1 = Asset(
        asset_uuid="C1",
        asset_type="system",
        name="C1",
        depth=1,
        measured=True,
        parent_uuid="P0",
    )
    l2 = Asset(
        asset_uuid="L2",
        asset_type="system",
        name="L2",
        depth=2,
        measured=True,
        parent_uuid="C1",
    )

    # distribution_substation chain (longer)
    ps0 = Asset(
        asset_uuid="PS0",
        asset_type="distribution_substation",
        name="PS0",
        depth=0,
        measured=True,
    )
    cs1 = Asset(
        asset_uuid="CS1",
        asset_type="distribution_substation",
        name="CS1",
        depth=1,
        measured=True,
        parent_uuid="PS0",
    )
    gs2 = Asset(
        asset_uuid="GS2",
        asset_type="distribution_substation",
        name="GS2",
        depth=2,
        measured=True,
        parent_uuid="CS1",
    )
    ls3 = Asset(
        asset_uuid="LS3",
        asset_type="distribution_substation",
        name="LS3",
        depth=3,
        measured=True,
        parent_uuid="GS2",
    )

    in_memory_session.add_all([p0, c1, l2, ps0, cs1, gs2, ls3])
    in_memory_session.flush()

    io = AssetsIO(in_memory_session)

    assert io.get_max_depth_non_leaf(asset_type="system") == 1
    assert io.get_max_depth_non_leaf(asset_type="distribution_substation") == 2
