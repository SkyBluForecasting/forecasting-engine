from unittest.mock import patch, MagicMock


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
@patch("forecasting_engine.tasks.fetch_weather_historical.OpenMeteoClient")
def test_main_success(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_historical.return_value = MagicMock(
        empty=False,
        to_csv=lambda index=False: "csv",
        to_json=lambda **kwargs: "json",
        to_string=lambda index=False: "text",
        __len__=lambda self: 2,
    )

    args = MagicMock(
        latitude=52.5, longitude=13.4, start_date=None, end_date=None, format="text"
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 0
        mock_logger.info.assert_any_call(
            "Fetching historical data for lat=52.5, lon=13.4"
        )
        mock_instance.fetch_historical.assert_called_once()
        mock_logger.info.assert_any_call("Successfully retrieved 2 historical records.")


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
@patch("forecasting_engine.tasks.fetch_weather_historical.OpenMeteoClient")
def test_main_empty_historical(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_historical.return_value = MagicMock(empty=True)

    args = MagicMock(
        latitude=52.5, longitude=13.4, start_date=None, end_date=None, format="text"
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 1
        mock_logger.warning.assert_called_with("No historical data returned.")


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
@patch("forecasting_engine.tasks.fetch_weather_historical.OpenMeteoClient")
def test_main_value_error(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_historical.side_effect = ValueError("bad input")

    args = MagicMock(
        latitude=52.5, longitude=13.4, start_date=None, end_date=None, format="text"
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 2
        mock_logger.error.assert_called_with("Invalid input: bad input")


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
@patch("forecasting_engine.tasks.fetch_weather_historical.OpenMeteoClient")
def test_main_exception(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_historical.side_effect = Exception("fail")

    args = MagicMock(
        latitude=52.5, longitude=13.4, start_date=None, end_date=None, format="text"
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 3
        mock_logger.exception.assert_called()


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
def test_main_invalid_start_date_format(mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    args = MagicMock(
        latitude=52.5,
        longitude=13.4,
        start_date="bad-date",
        end_date=None,
        format="text",
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 2
        mock_logger.error.assert_called_with(
            "Invalid date format: bad-date. Use YYYY-MM-DD."
        )


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
def test_main_invalid_end_date_format(mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    args = MagicMock(
        latitude=52.5,
        longitude=13.4,
        start_date=None,
        end_date="bad-datetime",
        format="text",
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 2
        mock_logger.error.assert_called_with(
            "Invalid datetime format: bad-datetime. Use YYYY-MM-DD HH:MM:SS."
        )


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
@patch("forecasting_engine.tasks.fetch_weather_historical.OpenMeteoClient")
def test_main_csv_output(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_historical.return_value = MagicMock(
        empty=False, to_csv=lambda index=False: "csv", __len__=lambda self: 1
    )
    args = MagicMock(
        latitude=52.5, longitude=13.4, start_date=None, end_date=None, format="csv"
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 0
        mock_logger.info.assert_any_call(
            "Fetching historical data for lat=52.5, lon=13.4"
        )
        mock_logger.info.assert_any_call("Successfully retrieved 1 historical records.")


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
@patch("forecasting_engine.tasks.fetch_weather_historical.OpenMeteoClient")
def test_main_json_output(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_historical.return_value = MagicMock(
        empty=False, to_json=lambda **kwargs: "json", __len__=lambda self: 1
    )
    args = MagicMock(
        latitude=52.5, longitude=13.4, start_date=None, end_date=None, format="json"
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 0
        mock_logger.info.assert_any_call(
            "Fetching historical data for lat=52.5, lon=13.4"
        )
        mock_logger.info.assert_any_call("Successfully retrieved 1 historical records.")


@patch("forecasting_engine.tasks.fetch_weather_historical.logger")
@patch("forecasting_engine.tasks.fetch_weather_historical.OpenMeteoClient")
def test_main_logs_start_and_end_dates(mock_client, mock_logger):
    from forecasting_engine.tasks.fetch_weather_historical import main

    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.fetch_historical.return_value = MagicMock(
        empty=False, to_string=lambda index=False: "text", __len__=lambda self: 1
    )
    args = MagicMock(
        latitude=52.5,
        longitude=13.4,
        start_date="2026-02-01",
        end_date="2026-02-22 12:00:00",
        format="text",
    )
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        result = main()
        assert result == 0
        mock_logger.info.assert_any_call("Starting from 2026-02-01")
        mock_logger.info.assert_any_call("Ending at 2026-02-22 12:00:00+00:00")
