from unittest.mock import MagicMock, patch
import forecasting_engine.services.weather_service.main as weather_main


def test_main_wires_dependencies_and_runs_assignment():
    session = MagicMock()
    session_local = MagicMock(return_value=session)

    assets_io_instance = MagicMock()
    weather_sites_io_instance = MagicMock()
    manager_instance = MagicMock()

    with patch.object(weather_main, "SessionLocal", session_local), patch.object(
        weather_main, "AssetsIO", MagicMock(return_value=assets_io_instance)
    ) as AssetsIO_mock, patch.object(
        weather_main,
        "WeatherSitesIO",
        MagicMock(return_value=weather_sites_io_instance),
    ) as WeatherSitesIO_mock, patch.object(
        weather_main, "WeatherManager", MagicMock(return_value=manager_instance)
    ) as WeatherManager_mock:

        weather_main.main()

        # Session created
        session_local.assert_called_once_with()

        # IOs instantiated with same session
        AssetsIO_mock.assert_called_once_with(session)
        WeatherSitesIO_mock.assert_called_once_with(session)

        # Manager created with IOs and run
        WeatherManager_mock.assert_called_once_with(
            assets_io_instance, weather_sites_io_instance
        )
        manager_instance.assign_db_assets_weather_sites.assert_called_once_with()

        # Session closed
        session.close.assert_called_once_with()


def test_main_closes_session_even_if_assignment_raises():
    session = MagicMock()
    session_local = MagicMock(return_value=session)

    manager_instance = MagicMock()
    manager_instance.assign_db_assets_weather_sites.side_effect = RuntimeError("boom")

    with patch.object(weather_main, "SessionLocal", session_local), patch.object(
        weather_main, "AssetsIO", MagicMock()
    ), patch.object(weather_main, "WeatherSitesIO", MagicMock()), patch.object(
        weather_main, "WeatherManager", MagicMock(return_value=manager_instance)
    ):

        # main should re-raise (no except), but still close session in finally
        try:
            weather_main.main()
        except RuntimeError:
            pass

        session.close.assert_called_once_with()
