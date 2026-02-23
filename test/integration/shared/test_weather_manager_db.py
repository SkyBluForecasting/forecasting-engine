"""
DB integration tests for WeatherManager + DB

Uses a real Postgres test DB (test_session fixture) to test the WeatherManager.

NOTE: We stub OpenMeteoClient to keep tests deterministic and offline.
"""

from __future__ import annotations

import pytest

from forecasting_db.models import Asset, WeatherSite, WeatherActual, WeatherForecast
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.weather_site_io import WeatherSitesIO
from forecasting_engine.shared.weather_manager import WeatherManager
from forecasting_engine.db_io.weather_actual_io import WeatherActualIO
from forecasting_engine.db_io.weather_forecast_io import WeatherForecastIO
from test.integration.conftest import DummyOpenMeteoClient


# ----------------------------
# Helpers
# ----------------------------


def _site_ids(session) -> set[str]:
    return {s.weather_site_id for s in session.query(WeatherSite).all()}


def _assert_weather_tables_non_empty(session) -> None:
    assert (
        session.query(WeatherActual).count() > 0
    ), "Expected WeatherActual rows to be inserted"
    assert (
        session.query(WeatherForecast).count() > 0
    ), "Expected WeatherForecast rows to be inserted"


def _assert_weather_rows_linked_to_known_sites(
    session, known_site_ids: set[str]
) -> None:
    actual_site_ids = {
        r[0] for r in session.query(WeatherActual.weather_site_id).distinct().all()
    }
    forecast_site_ids = {
        r[0] for r in session.query(WeatherForecast.weather_site_id).distinct().all()
    }

    assert actual_site_ids.issubset(
        known_site_ids
    ), f"WeatherActual contains unknown site_ids: {actual_site_ids - known_site_ids}"
    assert forecast_site_ids.issubset(
        known_site_ids
    ), f"WeatherForecast contains unknown site_ids: {forecast_site_ids - known_site_ids}"


def _assert_no_duplicate_weather_actual_keys(session, site_ids: set[str]) -> None:
    """
    WeatherActual uniqueness: (weather_site_id, variable, timestamp)
    """
    for site_id in site_ids:
        rows = session.query(WeatherActual).filter_by(weather_site_id=site_id).all()
        keys = [(r.weather_site_id, r.variable, r.timestamp) for r in rows]
        assert len(keys) == len(
            set(keys)
        ), f"Duplicate WeatherActual keys for {site_id}"


def _assert_no_duplicate_weather_forecast_keys(session, site_ids: set[str]) -> None:
    """
    WeatherForecast uniqueness: (weather_site_id, variable, timestamp, issue_time)
    """
    for site_id in site_ids:
        rows = session.query(WeatherForecast).filter_by(weather_site_id=site_id).all()
        keys = [
            (r.weather_site_id, r.variable, r.timestamp, r.issue_time) for r in rows
        ]
        assert len(keys) == len(
            set(keys)
        ), f"Duplicate WeatherForecast keys for {site_id}"


def _assert_no_duplicate_weather_keys(session, site_ids: set[str]) -> None:
    _assert_no_duplicate_weather_actual_keys(session, site_ids)
    _assert_no_duplicate_weather_forecast_keys(session, site_ids)


# ----------------------------
# DB fixtures: IO + managers
# ----------------------------


@pytest.fixture
def assets_io(test_session):
    return AssetsIO(test_session)


@pytest.fixture
def weather_sites_io(test_session):
    return WeatherSitesIO(test_session)


@pytest.fixture
def weather_manager(assets_io, weather_sites_io):
    # Assignment-only manager
    return WeatherManager(assets_io=assets_io, weather_sites_io=weather_sites_io)


@pytest.fixture
def manager_full(test_session, weather_manager):
    """
    Full WeatherManager configured to fetch + store weather, using a stubbed client.
    """
    return WeatherManager(
        assets_io=weather_manager.assets_io,
        weather_sites_io=weather_manager.weather_sites_io,
        weather_actual_io=WeatherActualIO(test_session),
        weather_forecast_io=WeatherForecastIO(test_session),
        client=DummyOpenMeteoClient(),
    )


# ----------------------------
# DB fixtures: assets
# ----------------------------


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
    # Seed referenced site so FK passes
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


def test_assign_sites_same_cell_creates_one_site_and_assigns_both_assets(
    test_session,
    weather_manager,
    manager_full,
    assets_same_cell,
):
    # Act: assign sites
    stats = weather_manager.assign_db_assets_weather_sites()

    # Assert: stats
    assert stats["total"] == 2
    assert stats["assigned"] == 2
    assert stats["skipped"] == 0
    assert stats["failed"] == 0
    assert stats["site_upserted"] == 1

    # Assert: both assets now point at same derived site
    a1 = test_session.query(Asset).filter_by(asset_uuid="A1").one()
    a2 = test_session.query(Asset).filter_by(asset_uuid="A2").one()
    assert a1.weather_site_id is not None
    assert a1.weather_site_id == a2.weather_site_id

    # Assert: exactly one WeatherSite created and coords match derivation
    sites = test_session.query(WeatherSite).all()
    assert len(sites) == 1
    site = sites[0]

    expected_id, expected_lat, expected_lon = WeatherManager._derive_weather_site(
        a1.latitude,
        a1.longitude,
        WeatherManager.WEATHER_GRID_RESOLUTION,
    )
    assert site.weather_site_id == expected_id == a1.weather_site_id
    assert float(site.site_lat) == expected_lat
    assert float(site.site_long) == expected_lon

    # Act: fetch + store weather
    data_stats = manager_full.fetch_and_store_weather_data(forecast_hours=6)

    # Assert: weather stored and linked to site
    assert data_stats["total_sites"] == 1
    assert data_stats["failed"] == 0

    known_ids = {site.weather_site_id}
    _assert_weather_tables_non_empty(test_session)
    _assert_weather_rows_linked_to_known_sites(test_session, known_ids)
    _assert_no_duplicate_weather_keys(test_session, known_ids)


def test_assign_sites_skips_preassigned_and_ignores_missing_coords(
    test_session,
    weather_manager,
    manager_full,
    assets_mixed_coords_and_preassigned,
):
    a_with, a_assigned, a_no_coords = assets_mixed_coords_and_preassigned

    # Act: assign sites
    stats = weather_manager.assign_db_assets_weather_sites()

    # Assert: stats (only 2 assets have coords)
    assert stats["total"] == 2
    assert stats["assigned"] == 1
    assert stats["skipped"] == 1
    assert stats["failed"] == 0
    assert stats["site_upserted"] == 1

    # Assert: DB state
    refreshed_with = (
        test_session.query(Asset).filter_by(asset_uuid=a_with.asset_uuid).one()
    )
    refreshed_assigned = (
        test_session.query(Asset).filter_by(asset_uuid=a_assigned.asset_uuid).one()
    )
    refreshed_no_coords = (
        test_session.query(Asset).filter_by(asset_uuid=a_no_coords.asset_uuid).one()
    )

    assert refreshed_with.weather_site_id is not None
    assert refreshed_with.weather_site_id.startswith("cell_")
    assert refreshed_assigned.weather_site_id == "already_set"
    assert refreshed_no_coords.weather_site_id is None

    ids = _site_ids(test_session)
    assert ids == {"already_set", refreshed_with.weather_site_id}

    # Act: fetch + store weather
    data_stats = manager_full.fetch_and_store_weather_data(forecast_hours=6)

    # Assert
    assert data_stats["total_sites"] == 2
    assert data_stats["failed"] == 0

    _assert_weather_tables_non_empty(test_session)
    _assert_weather_rows_linked_to_known_sites(test_session, ids)
    _assert_no_duplicate_weather_keys(test_session, ids)


def test_assign_and_fetch_are_idempotent(
    test_session,
    weather_manager,
    manager_full,
    assets_same_cell,
):
    # Assign: first run
    stats1 = weather_manager.assign_db_assets_weather_sites()
    assert stats1["assigned"] == 2
    assert stats1["site_upserted"] == 1

    site_count_1 = test_session.query(WeatherSite).count()

    # Assign: second run is a no-op
    stats2 = weather_manager.assign_db_assets_weather_sites()
    assert stats2["assigned"] == 0
    assert stats2["skipped"] == 2
    assert stats2["site_upserted"] == 0
    assert test_session.query(WeatherSite).count() == site_count_1

    # Fetch/store: first run
    fetch1 = manager_full.fetch_and_store_weather_data(forecast_hours=6)
    assert fetch1["failed"] == 0

    ids = _site_ids(test_session)
    assert ids, "Expected at least one WeatherSite"

    actual_count_1 = test_session.query(WeatherActual).count()
    forecast_count_1 = test_session.query(WeatherForecast).count()
    assert actual_count_1 > 0
    assert forecast_count_1 > 0

    _assert_weather_rows_linked_to_known_sites(test_session, ids)
    _assert_no_duplicate_weather_keys(test_session, ids)

    # Fetch/store: second run should not add duplicates
    fetch2 = manager_full.fetch_and_store_weather_data(forecast_hours=6)
    assert fetch2["failed"] == 0

    actual_count_2 = test_session.query(WeatherActual).count()
    forecast_count_2 = test_session.query(WeatherForecast).count()

    assert actual_count_2 == actual_count_1
    assert forecast_count_2 == forecast_count_1

    _assert_no_duplicate_weather_keys(test_session, ids)
