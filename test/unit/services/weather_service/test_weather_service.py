import pytest
from unittest.mock import patch, MagicMock


@patch("forecasting_engine.services.weather_service.main.SessionLocal")
@patch("forecasting_engine.services.weather_service.main.AssetsIO")
@patch("forecasting_engine.services.weather_service.main.WeatherSitesIO")
@patch("forecasting_engine.services.weather_service.main.WeatherActualIO")
@patch("forecasting_engine.services.weather_service.main.WeatherForecastIO")
@patch("forecasting_engine.services.weather_service.main.WeatherManager")
@patch("forecasting_engine.services.weather_service.main.get_logger")
def test_main_happy_path(
    mock_logger,
    mock_manager,
    mock_weather_forecast_io,
    mock_weather_actual_io,
    mock_weather_sites_io,
    mock_assets_io,
    mock_sessionlocal,
):
    from forecasting_engine.services.weather_service.main import main

    mock_session = MagicMock()
    mock_sessionlocal.return_value = mock_session
    mock_manager_instance = MagicMock()
    mock_manager.return_value = mock_manager_instance

    main()

    mock_sessionlocal.assert_called_once()
    mock_assets_io.assert_called_once_with(mock_session)
    mock_weather_sites_io.assert_called_once_with(mock_session)
    mock_weather_actual_io.assert_called_once_with(mock_session)
    mock_weather_forecast_io.assert_called_once_with(mock_session)
    mock_manager.assert_called_once()
    mock_manager_instance.assign_db_assets_weather_sites.assert_called_once()
    mock_manager_instance.fetch_and_store_weather_data.assert_called_once()
    mock_session.close.assert_called_once()


@patch("forecasting_engine.services.weather_service.main.SessionLocal")
def test_main_session_closed_on_exception(mock_sessionlocal):
    from forecasting_engine.services.weather_service.main import main

    mock_session = MagicMock()
    mock_sessionlocal.return_value = mock_session

    with patch(
        "forecasting_engine.services.weather_service.main.AssetsIO",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception):
            main()
    mock_session.close.assert_called_once()
