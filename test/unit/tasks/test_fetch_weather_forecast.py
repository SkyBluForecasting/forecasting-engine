from unittest.mock import patch, MagicMock


@patch("forecasting_engine.tasks.fetch_weather_forecast.logger")
@patch("forecasting_engine.tasks.fetch_weather_forecast.OpenMeteoClient")
def test_main_success(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_forecast import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_forecast.return_value = MagicMock(
        empty=False,
        to_csv=lambda index=False: "csv",
        to_json=lambda **kwargs: "json",
        to_string=lambda index=False: "text",
        __len__=lambda self: 2,
    )

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(latitude=52.5, longitude=13.4, hours=60, format="text"),
    ):
        result = main()
        assert result == 0
        mock_logger.info.assert_any_call(
            "Fetching 60-hour forecast for lat=52.5, lon=13.4"
        )
        mock_instance.fetch_forecast.assert_called_once()
        mock_logger.info.assert_any_call("Successfully retrieved 2 forecast records.")


@patch("forecasting_engine.tasks.fetch_weather_forecast.logger")
@patch("forecasting_engine.tasks.fetch_weather_forecast.OpenMeteoClient")
def test_main_empty_forecast(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_forecast import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_forecast.return_value = MagicMock(empty=True)

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(latitude=52.5, longitude=13.4, hours=60, format="text"),
    ):
        result = main()
        assert result == 1
        mock_logger.warning.assert_called_with("No forecast data returned.")


@patch("forecasting_engine.tasks.fetch_weather_forecast.logger")
@patch("forecasting_engine.tasks.fetch_weather_forecast.OpenMeteoClient")
def test_main_value_error(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_forecast import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_forecast.side_effect = ValueError("bad input")

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(latitude=52.5, longitude=13.4, hours=60, format="text"),
    ):
        result = main()
        assert result == 2
        mock_logger.error.assert_called_with("Invalid input: bad input")


@patch("forecasting_engine.tasks.fetch_weather_forecast.logger")
@patch("forecasting_engine.tasks.fetch_weather_forecast.OpenMeteoClient")
def test_main_exception(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_forecast import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_forecast.side_effect = Exception("fail")

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(latitude=52.5, longitude=13.4, hours=60, format="text"),
    ):
        result = main()
        assert result == 3
        mock_logger.exception.assert_called()


@patch("forecasting_engine.tasks.fetch_weather_forecast.logger")
@patch("forecasting_engine.tasks.fetch_weather_forecast.OpenMeteoClient")
def test_main_csv_output(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_forecast import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_forecast.return_value = MagicMock(
        empty=False, to_csv=lambda index=False: "csv", __len__=lambda self: 1
    )
    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(latitude=52.5, longitude=13.4, hours=60, format="csv"),
    ):
        result = main()
        assert result == 0
        mock_logger.info.assert_any_call(
            "Fetching 60-hour forecast for lat=52.5, lon=13.4"
        )
        mock_logger.info.assert_any_call("Successfully retrieved 1 forecast records.")


@patch("forecasting_engine.tasks.fetch_weather_forecast.logger")
@patch("forecasting_engine.tasks.fetch_weather_forecast.OpenMeteoClient")
def test_main_json_output(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_forecast import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_forecast.return_value = MagicMock(
        empty=False, to_json=lambda **kwargs: "json", __len__=lambda self: 1
    )
    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(latitude=52.5, longitude=13.4, hours=60, format="json"),
    ):
        result = main()
        assert result == 0
        mock_logger.info.assert_any_call(
            "Fetching 60-hour forecast for lat=52.5, lon=13.4"
        )
        mock_logger.info.assert_any_call("Successfully retrieved 1 forecast records.")
