import forecasting_engine.services.weather_service.main as svc_main

from forecasting_db.models import Asset, WeatherActual, WeatherForecast, WeatherSite
from test.integration.conftest import DummyOpenMeteoClient


def _assert_weather_unique_keys(session) -> None:
    # WeatherActual unique: (weather_site_id, variable, timestamp)
    actuals = session.query(WeatherActual).all()
    actual_keys = {(r.weather_site_id, r.variable, r.timestamp) for r in actuals}
    assert len(actual_keys) == len(actuals), "Duplicate WeatherActual unique keys found"

    # WeatherForecast unique: (weather_site_id, variable, timestamp, issue_time)
    forecasts = session.query(WeatherForecast).all()
    forecast_keys = {
        (r.weather_site_id, r.variable, r.timestamp, r.issue_time) for r in forecasts
    }
    assert len(forecast_keys) == len(
        forecasts
    ), "Duplicate WeatherForecast unique keys found"


def test_weather_service_main_runs_against_db(test_session, monkeypatch):
    """
    Smoke/integration test:
    - runs the real weather_service.main() against the integration test DB
    - verifies assignment + persistence happened
    - stays offline by patching OpenMeteoClient to DummyOpenMeteoClient
    """

    # --- Patch OpenMeteoClient used by the service ---
    monkeypatch.setattr(
        "forecasting_engine.shared.weather_manager.OpenMeteoClient",
        DummyOpenMeteoClient,
    )

    # --- Patch SessionLocal to use our test_session ---
    monkeypatch.setattr(svc_main, "SessionLocal", lambda: test_session)

    # --- Seed an asset that needs assignment ---
    test_session.add(
        Asset(
            asset_uuid="A1",
            asset_type="distribution_substation",
            name="A1",
            depth=0,
            measured=True,
            latitude=43.691,
            longitude=-79.299,
            weather_site_id=None,
        )
    )
    test_session.commit()

    # --- Run service ---
    svc_main.main()

    # --- Assert asset got assigned ---
    a1 = test_session.query(Asset).filter_by(asset_uuid="A1").one()
    assert a1.weather_site_id is not None
    assert a1.weather_site_id.startswith("cell_")

    # --- Assert WeatherSite exists ---
    site = (
        test_session.query(WeatherSite)
        .filter_by(weather_site_id=a1.weather_site_id)
        .one()
    )
    assert site.site_lat is not None
    assert site.site_long is not None

    # --- Assert weather persisted ---
    assert test_session.query(WeatherActual).count() > 0
    assert test_session.query(WeatherForecast).count() > 0

    # All rows should link to known sites (at least this one)
    known_site_ids = {
        r[0] for r in test_session.query(WeatherSite.weather_site_id).all()
    }
    actual_site_ids = {
        r[0] for r in test_session.query(WeatherActual.weather_site_id).distinct().all()
    }
    forecast_site_ids = {
        r[0]
        for r in test_session.query(WeatherForecast.weather_site_id).distinct().all()
    }
    assert actual_site_ids.issubset(known_site_ids)
    assert forecast_site_ids.issubset(known_site_ids)

    _assert_weather_unique_keys(test_session)


def test_weather_service_main_is_idempotent(test_session, monkeypatch):
    """
    Runs main() twice and asserts counts do not change due to ON CONFLICT DO NOTHING
    and stable DummyOpenMeteoClient issue_time.
    """

    monkeypatch.setattr(
        "forecasting_engine.shared.weather_manager.OpenMeteoClient",
        DummyOpenMeteoClient,
    )
    monkeypatch.setattr(svc_main, "SessionLocal", lambda: test_session)

    test_session.add(
        Asset(
            asset_uuid="A1",
            asset_type="distribution_substation",
            name="A1",
            depth=0,
            measured=True,
            latitude=43.691,
            longitude=-79.299,
            weather_site_id=None,
        )
    )
    test_session.commit()

    svc_main.main()
    actual_1 = test_session.query(WeatherActual).count()
    forecast_1 = test_session.query(WeatherForecast).count()
    assert actual_1 > 0
    assert forecast_1 > 0

    svc_main.main()
    actual_2 = test_session.query(WeatherActual).count()
    forecast_2 = test_session.query(WeatherForecast).count()

    assert actual_2 == actual_1
    assert forecast_2 == forecast_1
    _assert_weather_unique_keys(test_session)
