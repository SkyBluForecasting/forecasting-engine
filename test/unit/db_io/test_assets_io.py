import pytest
import pandas as pd


def uuids(rows):
    return [r.asset_uuid for r in rows]


# ----------------------------
# list_assets / get_asset
# ----------------------------


def test_list_assets_returns_all(asset_factory, assets_io):
    asset_factory(asset_uuid="A1", asset_type="system")
    asset_factory(asset_uuid="A2", asset_type="distribution_substation")

    assert set(uuids(assets_io.list_assets())) == {"A1", "A2"}


@pytest.mark.parametrize(
    "filter_type, expected",
    [
        ("system", ["A1"]),
        ("distribution_substation", ["A2"]),
    ],
)
def test_list_assets_filters_by_type(asset_factory, assets_io, filter_type, expected):
    asset_factory(asset_uuid="A1", asset_type="system")
    asset_factory(asset_uuid="A2", asset_type="distribution_substation")

    assert uuids(assets_io.list_assets(asset_type=filter_type)) == expected


def test_get_asset_found(asset_factory, assets_io):
    asset_factory(asset_uuid="A1")
    assert assets_io.get_asset("A1").asset_uuid == "A1"


def test_get_asset_not_found(assets_io):
    assert assets_io.get_asset("NOPE") is None


# ----------------------------
# to_df / from_df
# ----------------------------


def test_to_df_returns_dataframe(asset_factory, assets_io):
    asset_factory(asset_uuid="A1", name="Asset 1", asset_type="system", capacity_kw=100)

    df = assets_io.to_df()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert set(df.columns) == {
        "asset_uuid",
        "name",
        "asset_type",
        "capacity_kw",
        "parent_uuid",
    }


def test_to_df_empty_returns_empty_df(assets_io):
    assert assets_io.to_df().empty


def test_to_df_filters_by_type(asset_factory, assets_io):
    asset_factory(asset_uuid="A1", asset_type="system")
    asset_factory(asset_uuid="A2", asset_type="distribution_substation")

    df = assets_io.to_df(asset_type="distribution_substation")
    assert df["asset_uuid"].tolist() == ["A2"]


def test_from_df_not_implemented(assets_io):
    with pytest.raises(NotImplementedError, match="from_df"):
        assets_io.from_df(pd.DataFrame())


# ----------------------------
# leaf / non-leaf helpers
# ----------------------------


def build_chain(asset_factory, prefix, asset_type="system", depth0=0):
    """
    prefix0 -> prefix1 -> prefix2
    Returns (p0, c1, l2)
    """
    p0 = asset_factory(asset_uuid=f"{prefix}0", asset_type=asset_type, depth=depth0)
    c1 = asset_factory(
        asset_uuid=f"{prefix}1",
        asset_type=asset_type,
        depth=depth0 + 1,
        parent_uuid=p0.asset_uuid,
    )
    l2 = asset_factory(
        asset_uuid=f"{prefix}2",
        asset_type=asset_type,
        depth=depth0 + 2,
        parent_uuid=c1.asset_uuid,
    )
    return p0, c1, l2


# ----------------------------
# list_leaf_assets
# ----------------------------


def test_list_leaf_assets_basic(asset_factory, assets_io):
    # PARENT -> CHILD -> LEAF
    build_chain(asset_factory, "N")
    assert uuids(assets_io.list_leaf_assets()) == ["N2"]


def test_list_leaf_assets_top_level_leaf(asset_factory, assets_io):
    asset_factory(asset_uuid="LEAF", parent_uuid=None, depth=0)
    assert uuids(assets_io.list_leaf_assets()) == ["LEAF"]


def test_list_leaf_assets_multiple_leaves(asset_factory, assets_io):
    # Chain leaf + standalone leaf
    build_chain(asset_factory, "A")
    asset_factory(
        asset_uuid="B0", asset_type="distribution_substation", depth=0, parent_uuid=None
    )
    assert set(uuids(assets_io.list_leaf_assets())) == {"A2", "B0"}


def test_list_leaf_assets_filter_by_type(asset_factory, assets_io):
    build_chain(asset_factory, "SYS", asset_type="system")
    asset_factory(asset_uuid="SUB0", asset_type="distribution_substation", depth=0)
    assert uuids(assets_io.list_leaf_assets(asset_type="system")) == ["SYS2"]


# ----------------------------
# list_non_leaf_assets
# ----------------------------


def test_list_non_leaf_assets_basic(asset_factory, assets_io):
    p0, c1, _ = build_chain(asset_factory, "X")
    assert set(uuids(assets_io.list_non_leaf_assets())) == {
        p0.asset_uuid,
        c1.asset_uuid,
    }


def test_list_non_leaf_assets_no_children_returns_empty(asset_factory, assets_io):
    asset_factory(asset_uuid="A1")
    asset_factory(asset_uuid="A2")
    assert assets_io.list_non_leaf_assets() == []


def test_list_non_leaf_assets_filter_by_type(asset_factory, assets_io):
    build_chain(asset_factory, "SYS", asset_type="system")
    build_chain(asset_factory, "SUB", asset_type="distribution_substation")

    assert set(uuids(assets_io.list_non_leaf_assets(asset_type="system"))) == {
        "SYS0",
        "SYS1",
    }


# ----------------------------
# list_non_leaf_assets_at_depth
# ----------------------------


def test_list_non_leaf_assets_at_depth_returns_only_matching_depth(
    asset_factory, assets_io
):
    build_chain(asset_factory, "D")

    assert uuids(assets_io.list_non_leaf_assets_at_depth(0)) == ["D0"]
    assert uuids(assets_io.list_non_leaf_assets_at_depth(1)) == ["D1"]
    assert assets_io.list_non_leaf_assets_at_depth(2) == []


def test_list_non_leaf_assets_at_depth_orders_by_uuid(asset_factory, assets_io):
    # P0 -> A1 -> A2 and P0 -> B1 -> B2
    p0 = asset_factory(asset_uuid="P0", depth=0)
    a1 = asset_factory(asset_uuid="A1", depth=1, parent_uuid=p0.asset_uuid)
    b1 = asset_factory(asset_uuid="B1", depth=1, parent_uuid=p0.asset_uuid)
    asset_factory(asset_uuid="A2", depth=2, parent_uuid=a1.asset_uuid)
    asset_factory(asset_uuid="B2", depth=2, parent_uuid=b1.asset_uuid)

    assert uuids(assets_io.list_non_leaf_assets_at_depth(1)) == ["A1", "B1"]


def test_list_non_leaf_assets_at_depth_filter_by_type(asset_factory, assets_io):
    p0 = asset_factory(asset_uuid="P0", asset_type="system", depth=0)
    sub1 = asset_factory(
        asset_uuid="SUB1",
        asset_type="distribution_substation",
        depth=1,
        parent_uuid=p0.asset_uuid,
    )
    asset_factory(
        asset_uuid="SUB2",
        asset_type="distribution_substation",
        depth=2,
        parent_uuid=sub1.asset_uuid,
    )

    sys1 = asset_factory(
        asset_uuid="SYS1", asset_type="system", depth=1, parent_uuid=p0.asset_uuid
    )
    asset_factory(
        asset_uuid="SYS2", asset_type="system", depth=2, parent_uuid=sys1.asset_uuid
    )

    assert uuids(
        assets_io.list_non_leaf_assets_at_depth(1, asset_type="distribution_substation")
    ) == ["SUB1"]


# ----------------------------
# get_max_depth_non_leaf
# ----------------------------


def test_get_max_depth_non_leaf_basic(asset_factory, assets_io):
    build_chain(asset_factory, "M")
    assert assets_io.get_max_depth_non_leaf() == 1


def test_get_max_depth_non_leaf_empty_returns_zero(asset_factory, assets_io):
    asset_factory(asset_uuid="LEAF", depth=5)  # no children anywhere
    assert assets_io.get_max_depth_non_leaf() == 0


def test_get_max_depth_non_leaf_filter_by_type(asset_factory, assets_io):
    build_chain(asset_factory, "SYS", asset_type="system")  # max non-leaf depth 1
    # Longer chain: depths 0,1,2 are non-leaf => max 2
    ps0 = asset_factory(asset_uuid="PS0", asset_type="distribution_substation", depth=0)
    cs1 = asset_factory(
        asset_uuid="CS1",
        asset_type="distribution_substation",
        depth=1,
        parent_uuid=ps0.asset_uuid,
    )
    gs2 = asset_factory(
        asset_uuid="GS2",
        asset_type="distribution_substation",
        depth=2,
        parent_uuid=cs1.asset_uuid,
    )
    asset_factory(
        asset_uuid="LS3",
        asset_type="distribution_substation",
        depth=3,
        parent_uuid=gs2.asset_uuid,
    )

    assert assets_io.get_max_depth_non_leaf(asset_type="system") == 1
    assert assets_io.get_max_depth_non_leaf(asset_type="distribution_substation") == 2


# ----------------------------
# update_weather_site_id
# ----------------------------


def test_update_weather_site_id_sets_value_and_commits(asset_factory, assets_io):
    asset_factory(asset_uuid="A1", weather_site_id=None)

    assets_io.update_weather_site_id("A1", "SITE_123")
    assert assets_io.get_asset("A1").weather_site_id == "SITE_123"


def test_update_weather_site_id_raises_if_asset_missing(assets_io):
    with pytest.raises(ValueError, match="Asset DOES_NOT_EXIST not found"):
        assets_io.update_weather_site_id("DOES_NOT_EXIST", "SITE_123")


# ----------------------------
# list_assets_with_coords
# ----------------------------


def test_list_assets_with_coords_returns_only_assets_with_lat_lon(
    asset_factory, assets_io
):
    asset_factory(asset_uuid="A_WITH", latitude=43.0, longitude=-79.0)
    asset_factory(asset_uuid="A_NOLAT", latitude=None, longitude=-79.0)
    asset_factory(asset_uuid="A_NOLON", latitude=43.0, longitude=None)

    assert uuids(assets_io.list_assets_with_coords()) == ["A_WITH"]


def test_list_assets_with_coords_filters_by_type(asset_factory, assets_io):
    asset_factory(
        asset_uuid="SYS1", asset_type="system", latitude=43.1, longitude=-79.1
    )
    asset_factory(
        asset_uuid="SUB1",
        asset_type="distribution_substation",
        latitude=43.2,
        longitude=-79.2,
    )

    assert uuids(
        assets_io.list_assets_with_coords(asset_type="distribution_substation")
    ) == ["SUB1"]


@pytest.mark.parametrize(
    "missing_only, expected", [(True, ["A_MISSING"]), (False, {"A_MISSING", "A_HAS"})]
)
def test_list_assets_with_coords_missing_weather_site_only(
    asset_factory, assets_io, missing_only, expected
):
    asset_factory(
        asset_uuid="A_MISSING", latitude=43.3, longitude=-79.3, weather_site_id=None
    )
    asset_factory(
        asset_uuid="A_HAS", latitude=43.4, longitude=-79.4, weather_site_id="SITE_X"
    )

    result = assets_io.list_assets_with_coords(missing_weather_site_only=missing_only)
    got = uuids(result)

    if missing_only:
        assert got == expected
    else:
        assert set(got) == expected


def test_list_assets_with_coords_type_filter_and_missing_weather_site_only_combined(
    asset_factory, assets_io
):
    asset_factory(
        asset_uuid="SYS_MISSING",
        asset_type="system",
        latitude=43.5,
        longitude=-79.5,
        weather_site_id=None,
    )
    asset_factory(
        asset_uuid="SUB_MISSING",
        asset_type="distribution_substation",
        latitude=43.6,
        longitude=-79.6,
        weather_site_id=None,
    )
    asset_factory(
        asset_uuid="SYS_HAS",
        asset_type="system",
        latitude=43.7,
        longitude=-79.7,
        weather_site_id="SITE_Y",
    )

    assert uuids(
        assets_io.list_assets_with_coords(
            asset_type="system", missing_weather_site_only=True
        )
    ) == ["SYS_MISSING"]
