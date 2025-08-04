"""S3 loading and saving utility helpers"""

import os
import boto3
import pandas as pd
from forecasting_engine.orchestration.logger_factory import get_logger
from io import StringIO
from dotenv import load_dotenv
from pandas.errors import EmptyDataError

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def get_s3_bucket():
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is not set.")
    return bucket


def get_s3_client():
    return boto3.client("s3")


def extract_asset_id_from_s3_key(s3_key: str) -> str:
    filename = os.path.basename(s3_key)  # e.g. "L9M_train.csv" or "L9M.csv"
    filename_no_ext = os.path.splitext(filename)[0]  # Remove file extension first
    asset_id = filename_no_ext.split("_")[0]  # Then split on underscore
    return asset_id


def find_matching_key(asset_id: str, prefix: str) -> str:
    """
    Searches S3 for a key that contains the asset_id in the file name under the given prefix.

    Returns the first matching key found. If multiple are found, it logs a warning.

    Raises:
        ValueError: If S3 BUCKET environment variable is unset.
        FileNotFoundError: If no matching key is found.
    """
    bucket = get_s3_bucket()
    s3_client = get_s3_client()

    paginator = s3_client.get_paginator("list_objects_v2")
    matches = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if asset_id in key and key.endswith(".csv"):
                matches.append(key)

    if not matches:
        raise FileNotFoundError(
            f"No CSV file containing '{asset_id}' found under '{prefix}' in bucket {bucket}"
        )

    if len(matches) > 1:
        logger.warning(
            f"Multiple files found for asset_id '{asset_id}' under '{prefix}': {matches}"
        )

    logger.info(f"Using S3 key: {matches[0]}")
    return matches[0]


def load_training_csv_from_s3(key: str) -> pd.DataFrame:
    """
    Loads and parses a CSV from S3 into a pandas DataFrame.

    Args:
        key: Full S3 key of the file to download.

    Returns:
        DataFrame with datetime index and cleaned columns.

    Raises:
        ValueError: If S3 BUCKET environment variable is unset.
        ValueError: If the CSV file is empty
        ValueError: If the CSV is missing datetime or load column

    """

    S3_BUCKET = get_s3_bucket()
    s3_client = get_s3_client()

    logger.info(f"Loading from S3: {S3_BUCKET}/{key}")
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    csv_content = response["Body"].read().decode("utf-8")

    try:
        df = pd.read_csv(StringIO(csv_content))
    except EmptyDataError:
        raise ValueError(f"CSV at {S3_BUCKET}/{key} is empty")

    df.columns = df.columns.str.strip()
    # Check for required columns
    if "datetime" not in df.columns:
        raise ValueError(f"CSV at {S3_BUCKET}/{key} must contain a 'datetime' column.")
    if "load" not in df.columns:
        raise ValueError(f"CSV at {S3_BUCKET}/{key} must contain a 'load' column.")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    # Ensure 'load' is the first column
    cols = ["load"] + [c for c in df.columns if c != "load"]
    df = df[cols]

    return df


def load_training_pd_from_s3(asset_id: str) -> pd.DataFrame:
    """
    Finds a CSV file containing asset_id in the 'training/' folder,
    loads it into a DataFrame, and returns it.

    Args:
        asset_id: ID of the asset to search for.

    Returns:
        Cleaned pandas DataFrame with datetime index.

    Raises:
        FileNotFoundError if no matching file is found.
        ValueError: If S3 BUCKET environment variable is unset.
        ValueError: If the CSV file is empty
        ValueError: If the CSV is missing datetime or load column
    """
    s3_key = find_matching_key(asset_id=asset_id, prefix="training/")
    return load_training_csv_from_s3(s3_key)


def save_forecast_csv_to_s3(df: pd.DataFrame, asset_id: str) -> None:
    """
    Saves a forecast DataFrame as a CSV file to S3 under:
    forecasts/{asset_id}/{start}_{end}.csv

    Args:
        df (pd.DataFrame): Forecast DataFrame with a datetime index.
        asset_id (str): ID of the asset the forecast belongs to.

    Raises:
        ValueError: If S3_BUCKET is not set.
    """

    S3_BUCKET = get_s3_bucket()
    s3_client = get_s3_client()

    # Format start and end timestamps as YYYYMMDDTHHMM
    start_str = df.index[0].strftime("%Y%m%dT%H%M")
    end_str = df.index[-1].strftime("%Y%m%dT%H%M")

    key = f"forecasts/{asset_id}/{start_str}_{end_str}.csv"

    # Convert to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=True)

    logger.info(f"Saving forecast to S3: {S3_BUCKET}/{key}")
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=csv_buffer.getvalue())
    logger.info("Forecast CSV successfully saved to S3.")
