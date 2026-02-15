import json

from forecasting_engine.config import SQS_CLIENT, QUEUE_URL
from forecasting_engine.db_io.measurement_io import MeasurementsIO
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_engine.db_io.forecast_run_io import ForecastRunIO
from forecasting_engine.db_io.session import SessionLocal
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_assets_to_forecast(measurement_io, forecast_run_io, assets_io):
    """
    Determine which non-pv leaf-node assets require a new forecast based on their latest measurements.


    Steps:
      1. Retrieve all non-pv leaf-node assets (assets with no children) from the database
      2. Retrieve the latest measurement timestamp for each asset.
      3. Retrieve the most recent forecast run timestamp for the same set of assets.
      4. Compare timestamps — if a measurement is newer than the last forecast,
         that asset needs a new forecast run.

    Returns:
        dict[str, datetime]: A mapping of leaf asset_id -> latest measurement timestamp
                             for assets that require a forecast update.

    """

    leaf_assets = assets_io.list_leaf_assets()

    non_pv_leaf_asset_ids = [a.asset_uuid for a in leaf_assets if a.asset_type != "pv"]

    if not non_pv_leaf_asset_ids:
        return {}

    latest_loads = measurement_io.get_latest_load_per_asset(
        asset_uuids=non_pv_leaf_asset_ids
    )
    if not latest_loads:
        return {}

    latest_forecasts = forecast_run_io.get_latest_forecast_run_start_time_per_asset(
        list(latest_loads.keys())
    )

    assets_to_forecast = {}
    for asset_id, (ts_last_measurement, _) in latest_loads.items():
        last_forecast_time = latest_forecasts.get(asset_id)
        if not last_forecast_time or ts_last_measurement > last_forecast_time:
            assets_to_forecast[asset_id] = ts_last_measurement

    return assets_to_forecast


def enqueue_assets(assets_to_forecast):
    """Send SQS messages for assets that need forecasts"""
    for asset_id, ts_last_measurement in assets_to_forecast.items():
        message = {"asset_id": asset_id}
        SQS_CLIENT.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(message))
        logger.warning(
            f"Enqueued forecast for asset {asset_id} (latest measurement: {ts_last_measurement})"
        )


def enqueue_new_forecasts():
    """Main function to enqueue new forecasts for all assets"""
    with SessionLocal() as session:
        measurement_io = MeasurementsIO(session)
        forecast_run_io = ForecastRunIO(session)
        assets_io = AssetsIO(session)

        assets_to_forecast = get_assets_to_forecast(
            measurement_io, forecast_run_io, assets_io
        )
        logger.warning(assets_to_forecast)
        if not assets_to_forecast:
            logger.warning("No assets require a forecast")
            return

        enqueue_assets(assets_to_forecast)
