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
    weather_site_id, site_lat, site_long = WeatherManager._derive_weather_site(
        43.69107389763945, -79.29900232909475
    )
    assert weather_site_id == "cell_873_-1586"
    assert site_lat == 43.675
    assert site_long == -79.275


def test_derive_weather_site_uses_custom_resolution():
    weather_site_id, site_lat, site_long = WeatherManager._derive_weather_site(
        43.691, -79.299, res=0.10
    )
    assert weather_site_id == "cell_436_-793"
    assert site_lat == 43.65
    assert site_long == -79.25


def test_derive_weather_site_negative_coords():
    weather_site_id, site_lat, site_long = WeatherManager._derive_weather_site(
        -43.691, -79.299
    )
    assert weather_site_id.startswith("cell_")
    assert isinstance(site_lat, float)
    assert isinstance(site_long, float)


def test_derive_weather_site_zero_coords():
    weather_site_id, site_lat, site_long = WeatherManager._derive_weather_site(0.0, 0.0)
    assert weather_site_id == "cell_0_0"
    assert site_lat == 0.025
    assert site_long == 0.025


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


def test_weather_manager_constructor_defaults():
    assets_io = MagicMock()
    weather_sites_io = MagicMock()
    manager = WeatherManager(assets_io=assets_io, weather_sites_io=weather_sites_io)
    assert manager.assets_io is assets_io
    assert manager.weather_sites_io is weather_sites_io
    assert manager.weather_actual_io is None
    assert manager.weather_forecast_io is None
    assert manager.client is not None
    assert manager.WEATHER_GRID_RESOLUTION == 0.05


def test_assign_db_assets_weather_sites_missing_lat_lon(
    manager, assets_io, weather_sites_io
):
    a1 = SimpleNamespace(
        asset_uuid="A1", latitude=None, longitude=-79.299, weather_site_id=None
    )
    a2 = SimpleNamespace(
        asset_uuid="A2", latitude=43.691, longitude=None, weather_site_id=None
    )
    assets_io.list_assets_with_coords.return_value = [a1, a2]
    stats = manager.assign_db_assets_weather_sites()
    assert stats["failed"] == 2
    assets_io.update_weather_site_id.assert_not_called()


def test_assign_db_assets_weather_sites_asset_missing_fields(
    manager, assets_io, weather_sites_io
):
    a1 = SimpleNamespace(asset_uuid="A1")  # missing latitude/longitude
    assets_io.list_assets_with_coords.return_value = [a1]
    stats = manager.assign_db_assets_weather_sites()
    assert stats["failed"] == 1
    assets_io.update_weather_site_id.assert_not_called()


# ----------------------------
# fetch_and_store_weather_data tests
# ----------------------------


@pytest.fixture
def weather_actual_io():
    return MagicMock()


@pytest.fixture
def weather_forecast_io():
    return MagicMock()


@pytest.fixture
def client():
    return MagicMock()


@pytest.fixture
def full_manager(
    assets_io, weather_sites_io, weather_actual_io, weather_forecast_io, client
):
    return WeatherManager(
        assets_io=assets_io,
        weather_sites_io=weather_sites_io,
        weather_actual_io=weather_actual_io,
        weather_forecast_io=weather_forecast_io,
        client=client,
    )


def test_fetch_and_store_weather_data_no_io_returns_empty_stats(full_manager):
    m = WeatherManager(MagicMock(), MagicMock())
    stats = m.fetch_and_store_weather_data()
    assert stats == {
        "total_sites": 0,
        "forecast_records": 0,
        "historical_records": 0,
        "failed": 0,
    }


def test_fetch_and_store_weather_data_success(
    full_manager, weather_sites_io, client, weather_forecast_io, weather_actual_io
):
    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    weather_sites_io.list_sites.return_value = [site]
    full_manager._fetch_and_store_forecast_for_site = MagicMock(return_value=5)
    full_manager._fetch_and_store_historical_for_site = MagicMock(return_value=3)
    stats = full_manager.fetch_and_store_weather_data(forecast_hours=24)
    assert stats["total_sites"] == 1
    assert stats["forecast_records"] == 5
    assert stats["historical_records"] == 3
    assert stats["failed"] == 0


def test_fetch_and_store_weather_data_handles_failure(full_manager, weather_sites_io):
    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    weather_sites_io.list_sites.return_value = [site]
    full_manager._fetch_and_store_forecast_for_site = MagicMock(
        side_effect=Exception("fail")
    )
    full_manager._fetch_and_store_historical_for_site = MagicMock(return_value=0)
    stats = full_manager.fetch_and_store_weather_data()
    assert stats["failed"] == 1


def test_fetch_and_store_forecast_for_site_empty_df(full_manager, client):
    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    client.fetch_forecast.return_value = MagicMock(empty=True)
    result = full_manager._fetch_and_store_forecast_for_site(site, forecast_hours=24)
    assert result == 0


def test_fetch_and_store_forecast_for_site_inserts_records(
    full_manager, client, weather_forecast_io
):
    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    df = MagicMock(empty=False, copy=lambda: df)
    df.__getitem__.side_effect = lambda key: (
        [1, 2, 3] if key == "weather_site_id" else ["a", "b", "c"]
    )
    df.to_dict.return_value = [{}, {}, {}]
    client.fetch_forecast.return_value = df
    weather_forecast_io.bulk_insert_forecasts.return_value = 3
    result = full_manager._fetch_and_store_forecast_for_site(site, forecast_hours=24)
    assert result == 3
    weather_forecast_io.bulk_insert_forecasts.assert_called_once()


def test_fetch_and_store_historical_for_site_empty_df(
    full_manager, client, weather_actual_io
):
    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    weather_actual_io.get_latest_reading_timestamps.return_value = {"temp_air_c": None}
    client.fetch_historical.return_value = MagicMock(empty=True)
    result = full_manager._fetch_and_store_historical_for_site(site)
    assert result == 0


def test_fetch_and_store_historical_for_site_inserts_records(
    full_manager, client, weather_actual_io
):
    import pandas as pd

    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    # Simulate latest readings
    weather_actual_io.get_latest_reading_timestamps.return_value = {
        "temp_air_c": pd.Timestamp("2026-02-20T10:00")
    }
    # Simulate historical data
    df = pd.DataFrame(
        {
            "weather_site_id": [site.weather_site_id, site.weather_site_id],
            "timestamp": [
                pd.Timestamp("2026-02-21T10:00"),
                pd.Timestamp("2026-02-22T10:00"),
            ],
            "variable": ["temp_air_c", "temp_air_c"],
            "value": [10.0, 12.0],
        }
    )
    client.fetch_historical.return_value = df
    weather_actual_io.bulk_insert_readings.return_value = 2
    result = full_manager._fetch_and_store_historical_for_site(site)
    assert result == 2
    weather_actual_io.bulk_insert_readings.assert_called_once()


def test_fetch_and_store_historical_for_site_inserts_zero_records(
    full_manager, client, weather_actual_io
):
    import pandas as pd

    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    weather_actual_io.get_latest_reading_timestamps.return_value = {"temp_air_c": None}
    df = pd.DataFrame(
        {
            "weather_site_id": [site.weather_site_id],
            "timestamp": [pd.Timestamp("2026-02-21T10:00")],
            "variable": ["temp_air_c"],
            "value": [10.0],
        }
    )
    client.fetch_historical.return_value = df
    weather_actual_io.bulk_insert_readings.return_value = 0
    result = full_manager._fetch_and_store_historical_for_site(site)
    assert result == 0
    weather_actual_io.bulk_insert_readings.assert_called_once()


def test_fetch_and_store_historical_for_site_no_variables_to_insert(
    full_manager, client, weather_actual_io
):
    import pandas as pd

    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    weather_actual_io.get_latest_reading_timestamps.return_value = {
        "temp_air_c": pd.Timestamp("2026-02-20T10:00")
    }
    # DataFrame with variable not matching WeatherVariable
    df = pd.DataFrame(
        {
            "weather_site_id": [site.weather_site_id],
            "timestamp": [pd.Timestamp("2026-02-21T10:00")],
            "variable": ["not_a_valid_var"],
            "value": [10.0],
        }
    )
    client.fetch_historical.return_value = df
    weather_actual_io.bulk_insert_readings.return_value = 0
    result = full_manager._fetch_and_store_historical_for_site(site)
    assert result == 0
    weather_actual_io.bulk_insert_readings.assert_not_called()


def test_fetch_and_store_historical_for_site_all_filtered_out(
    full_manager, client, weather_actual_io
):
    import pandas as pd

    site = SimpleNamespace(weather_site_id="cell_1_1", site_lat=43.0, site_long=-79.0)
    # Simulate latest readings
    weather_actual_io.get_latest_reading_timestamps.return_value = {
        "temp_air_c": pd.Timestamp("2026-02-22T10:00")
    }
    # Simulate historical data with timestamps before latest
    df = pd.DataFrame(
        {
            "weather_site_id": [site.weather_site_id],
            "timestamp": [pd.Timestamp("2026-02-21T10:00")],
            "variable": ["temp_air_c"],
            "value": [10.0],
        }
    )
    client.fetch_historical.return_value = df
    weather_actual_io.bulk_insert_readings.return_value = 0
    result = full_manager._fetch_and_store_historical_for_site(site)
    assert result == 0
    weather_actual_io.bulk_insert_readings.assert_not_called()
