"""DB integration tests for WeatherManager + AssetsIO + WeatherSitesIO.

Uses a real Postgres test DB (test_session fixture) to test the WeatherManager.
"""

import pytest

from forecasting_db.models import Asset, WeatherSite
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.weather_site_io import WeatherSitesIO
from forecasting_engine.shared.weather_manager import WeatherManager


# ----------------------------
# DB fixtures: assets
# ----------------------------


@pytest.fixture
def assets_io(test_session):
    return AssetsIO(test_session)


@pytest.fixture
def weather_sites_io(test_session):
    return WeatherSitesIO(test_session)


@pytest.fixture
def weather_manager(assets_io, weather_sites_io):
    return WeatherManager(assets_io=assets_io, weather_sites_io=weather_sites_io)


@pytest.fixture
def assets_same_cell(test_session):
    """
    Two assets that should derive the same weather cell id.
    (Make sure they're in the same 0.05 grid cell.)
    """
    a1 = Asset(
        asset_uuid="A1",
        asset_type="distribution_substation",
        name="A1",
        depth=0,
        measured=True,
        latitude=43.691,  # same cell as 43.692 at res=0.05
        longitude=-79.299,
        weather_site_id=None,
    )
    a2 = Asset(
        asset_uuid="A2",
        asset_type="distribution_substation",
        name="A2",
        depth=0,
        measured=True,
        latitude=43.692,
        longitude=-79.298,
        weather_site_id=None,
    )
    test_session.add_all([a1, a2])
    test_session.commit()
    return [a1, a2]


@pytest.fixture
def assets_mixed_coords_and_preassigned(test_session):
    """
    - A_WITH: has coords, unassigned
    - A_ASSIGNED: has coords, already assigned (should be skipped)
    - A_NO_COORDS: missing coords (should not be processed at all)
    """
    # Seed the referenced site so the FK passes
    test_session.add(
        WeatherSite(weather_site_id="already_set", site_lat=0.0, site_long=0.0)
    )
    test_session.commit()
    a_with = Asset(
        asset_uuid="A_WITH",
        asset_type="distribution_substation",
        name="A_WITH",
        depth=0,
        measured=True,
        latitude=43.691,
        longitude=-79.299,
        weather_site_id=None,
    )
    a_assigned = Asset(
        asset_uuid="A_ASSIGNED",
        asset_type="distribution_substation",
        name="A_ASSIGNED",
        depth=0,
        measured=True,
        latitude=43.691,
        longitude=-79.299,
        weather_site_id="already_set",
    )
    a_no_coords = Asset(
        asset_uuid="A_NO_COORDS",
        asset_type="distribution_substation",
        name="A_NO_COORDS",
        depth=0,
        measured=True,
        latitude=None,
        longitude=None,
        weather_site_id=None,
    )
    test_session.add_all([a_with, a_assigned, a_no_coords])
    test_session.commit()
    return a_with, a_assigned, a_no_coords


# ----------------------------
# Tests
# ----------------------------


def test_weather_manager_assigns_sites_and_persists_assets_and_sites(
    test_session,
    weather_manager,
    assets_same_cell,
):
    stats = weather_manager.assign_db_assets_weather_sites()

    # Only the two with coords exist, none pre-assigned
    assert stats["total"] == 2
    assert stats["assigned"] == 2
    assert stats["skipped"] == 0
    assert stats["failed"] == 0

    # Both assets should now have the same derived weather_site_id
    a1 = test_session.query(Asset).filter_by(asset_uuid="A1").one()
    a2 = test_session.query(Asset).filter_by(asset_uuid="A2").one()
    assert a1.weather_site_id is not None
    assert a1.weather_site_id == a2.weather_site_id

    # Only one WeatherSite row should exist (same cell)
    sites = test_session.query(WeatherSite).all()
    assert len(sites) == 1
    assert sites[0].weather_site_id == a1.weather_site_id

    # And site coords should match the manager’s derivation logic
    expected_id, expected_lat, expected_lon = WeatherManager._derive_weather_site(
        a1.latitude,
        a1.longitude,
        WeatherManager.WEATHER_GRID_RESOLUTION,
    )
    assert sites[0].weather_site_id == expected_id
    assert float(sites[0].site_lat) == expected_lat
    assert float(sites[0].site_long) == expected_lon

    # site_upserted: at least 1 (should be exactly 1 here since both same cell)
    assert stats["site_upserted"] == 1


def test_weather_manager_skips_preassigned_and_ignores_assets_missing_coords(
    test_session,
    weather_manager,
    assets_mixed_coords_and_preassigned,
):
    a_with, a_assigned, a_no_coords = assets_mixed_coords_and_preassigned

    stats = weather_manager.assign_db_assets_weather_sites()

    # list_assets_with_coords() should return ONLY the two that have coords
    assert stats["total"] == 2

    # one is assigned, one skipped
    assert stats["assigned"] == 1
    assert stats["skipped"] == 1
    assert stats["failed"] == 0
    assert stats["site_upserted"] == 1  # only the new one needs site creation

    refreshed_with = (
        test_session.query(Asset).filter_by(asset_uuid=a_with.asset_uuid).one()
    )
    refreshed_assigned = (
        test_session.query(Asset).filter_by(asset_uuid=a_assigned.asset_uuid).one()
    )
    refreshed_no_coords = (
        test_session.query(Asset).filter_by(asset_uuid=a_no_coords.asset_uuid).one()
    )

    # newly assigned got a derived site id
    assert refreshed_with.weather_site_id is not None
    assert refreshed_with.weather_site_id.startswith("cell_")

    # preassigned unchanged
    assert refreshed_assigned.weather_site_id == "already_set"

    # missing-coords should still be None (never processed)
    assert refreshed_no_coords.weather_site_id is None

    # should have 2 WeatherSite rows:
    #  - the preseeded "already_set and the derived one for A_WITH
    sites = test_session.query(WeatherSite).all()
    site_ids = {s.weather_site_id for s in sites}

    assert site_ids == {"already_set", refreshed_with.weather_site_id}


def test_weather_manager_is_idempotent_when_run_twice(
    test_session,
    weather_manager,
    assets_same_cell,
):
    stats1 = weather_manager.assign_db_assets_weather_sites()

    # first run assigns
    assert stats1["total"] == 2
    assert stats1["assigned"] == 2
    assert stats1["skipped"] == 0
    assert stats1["failed"] == 0
    assert stats1["site_upserted"] == 1

    site_count_1 = test_session.query(WeatherSite).count()

    # second run should skip (both already assigned), and NOT create new sites
    stats2 = weather_manager.assign_db_assets_weather_sites()

    assert stats2["total"] == 2
    assert stats2["assigned"] == 0
    assert stats2["skipped"] == 2
    assert stats2["failed"] == 0
    assert stats2["site_upserted"] == 0

    site_count_2 = test_session.query(WeatherSite).count()
    assert site_count_2 == site_count_1
