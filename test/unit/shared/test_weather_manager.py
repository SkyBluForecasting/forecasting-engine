import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from forecasting_engine.shared.weather_manager import WeatherManager


# ----------------------------
# _derive_weather_site tests
# ----------------------------


def test_derive_weather_site_raises_on_none_lat_lon():
    with pytest.raises(ValueError, match="must not be None"):
        WeatherManager._derive_weather_site(None, -79.0)

    with pytest.raises(ValueError, match="must not be None"):
        WeatherManager._derive_weather_site(43.0, None)


def test_derive_weather_site_deterministic_default_resolution():
    # res=0.05:
    # lat=43.691073... -> floor(43.691073/0.05)=floor(873.82146)=873 -> cell_lat=43.65
    # lon=-79.299002... -> floor(-79.299002/0.05)=floor(-1585.98004)=-1586 -> cell_lon=-79.30
    # center = cell + 0.025
    weather_site_id, site_lat, site_long = WeatherManager._derive_weather_site(
        43.69107389763945, -79.29900232909475
    )

    assert weather_site_id == "cell_43.65_-79.30"
    assert site_lat == 43.675
    assert site_long == -79.275


def test_derive_weather_site_uses_custom_resolution():
    weather_site_id, site_lat, site_long = WeatherManager._derive_weather_site(
        43.691, -79.299, res=0.10
    )

    # cell_lat = floor(43.691/0.1)*0.1 = 43.6
    # cell_lon = floor(-79.299/0.1)*0.1 = -79.3
    # center = +0.05
    assert weather_site_id == "cell_43.60_-79.30"
    assert site_lat == 43.65
    assert site_long == -79.25


# ----------------------------
# assign_db_assets_weather_sites tests
# ----------------------------


@pytest.fixture
def assets_io():
    return MagicMock()


@pytest.fixture
def weather_sites_io():
    return MagicMock()


@pytest.fixture
def manager(assets_io, weather_sites_io):
    return WeatherManager(assets_io=assets_io, weather_sites_io=weather_sites_io)


def test_assign_db_assets_weather_sites_assigns_skips_and_counts_upsert(
    manager, assets_io, weather_sites_io
):
    """
    assets:
      - A1: no weather_site_id -> assign, upsert True
      - A2: already has weather_site_id -> skipped
      - A3: no weather_site_id -> assign, upsert False (site existed)
    """
    a1 = SimpleNamespace(
        asset_uuid="A1", latitude=43.691, longitude=-79.299, weather_site_id=None
    )
    a2 = SimpleNamespace(
        asset_uuid="A2",
        latitude=43.691,
        longitude=-79.299,
        weather_site_id="existing_site",
    )
    a3 = SimpleNamespace(
        asset_uuid="A3", latitude=43.691, longitude=-79.299, weather_site_id=None
    )

    assets_io.list_assets_with_coords.return_value = [a1, a2, a3]

    # upsert True for A1's derived site, False for A3's derived site
    weather_sites_io.upsert_site.side_effect = [True, False]

    stats = manager.assign_db_assets_weather_sites()

    assert stats == {
        "total": 3,
        "assigned": 2,
        "skipped": 1,
        "failed": 0,
        "site_upserted": 1,
    }

    # Two assignments -> update called twice
    assert assets_io.update_weather_site_id.call_count == 2

    # ensure update called with correct asset uuids and derived site ids
    calls = assets_io.update_weather_site_id.call_args_list
    called_pairs = [(c.args[0], c.args[1]) for c in calls]

    # Both A1 and A3 should be set to the same site id given same coords
    assert called_pairs[0][0] == "A1"
    assert called_pairs[1][0] == "A3"
    assert called_pairs[0][1].startswith("cell_")
    assert called_pairs[1][1] == called_pairs[0][1]

    # upsert called twice (A1 and A3), not for skipped A2
    assert weather_sites_io.upsert_site.call_count == 2


def test_assign_db_assets_weather_sites_counts_failed_when_upsert_raises(
    manager, assets_io, weather_sites_io
):
    a1 = SimpleNamespace(
        asset_uuid="A1", latitude=43.691, longitude=-79.299, weather_site_id=None
    )
    assets_io.list_assets_with_coords.return_value = [a1]

    weather_sites_io.upsert_site.side_effect = RuntimeError("boom")

    stats = manager.assign_db_assets_weather_sites()

    assert stats == {
        "total": 1,
        "assigned": 0,
        "skipped": 0,
        "failed": 1,
        "site_upserted": 0,
    }

    assets_io.update_weather_site_id.assert_not_called()


def test_assign_db_assets_weather_sites_continues_after_failure(
    manager, assets_io, weather_sites_io
):
    a1 = SimpleNamespace(
        asset_uuid="A1", latitude=43.691, longitude=-79.299, weather_site_id=None
    )
    a2 = SimpleNamespace(
        asset_uuid="A2", latitude=43.692, longitude=-79.298, weather_site_id=None
    )
    assets_io.list_assets_with_coords.return_value = [a1, a2]

    # First upsert fails, second succeeds
    weather_sites_io.upsert_site.side_effect = [RuntimeError("boom"), True]

    stats = manager.assign_db_assets_weather_sites()

    assert stats["total"] == 2
    assert stats["failed"] == 1
    assert stats["assigned"] == 1
    assert stats["site_upserted"] == 1
    assert stats["skipped"] == 0

    # Only the second asset should have been updated
    assert assets_io.update_weather_site_id.call_count == 1
    assert assets_io.update_weather_site_id.call_args[0][0] == "A2"


def test_assign_db_assets_weather_sites_when_no_assets(manager, assets_io):
    assets_io.list_assets_with_coords.return_value = []

    stats = manager.assign_db_assets_weather_sites()

    assert stats == {
        "total": 0,
        "assigned": 0,
        "skipped": 0,
        "failed": 0,
        "site_upserted": 0,
    }
