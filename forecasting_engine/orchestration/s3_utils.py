"""S3 loading and saving utility helpers"""

import os
import boto3
import re
import pandas as pd
from forecasting_engine.orchestration.logger_factory import get_logger
from typing import List
from io import StringIO
from forecasting_engine.config import S3_BUCKET
from pandas.errors import EmptyDataError

logger = get_logger(__name__)

TRAINING_PREFIX = "training/"


def get_s3_client():
    return boto3.client("s3")


def find_matching_key(asset_id: str, prefix: str) -> str:
    """
    Searches S3 for a key that contains the asset_id in the file name under the given prefix.

    Returns the first matching key found. If multiple are found, it logs a warning.

    Raises:
        ValueError: If S3 BUCKET environment variable is unset.
        FileNotFoundError: If no matching key is found.
    """
    s3_client = get_s3_client()

    paginator = s3_client.get_paginator("list_objects_v2")
    matches = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if asset_id in key and key.endswith(".csv"):
                matches.append(key)

    if not matches:
        raise FileNotFoundError(
            f"No CSV file containing '{asset_id}' found under '{prefix}' in bucket {S3_BUCKET}"
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

    # Drop any non-numeric leftover columns (like FSA IDs)
    df = df.select_dtypes(include=["number", "bool"])

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


def list_training_fsa_ids() -> List[str]:
    S3_CLIENT = get_s3_client()

    """List FSA IDs under S3 prefix."""
    paginator = S3_CLIENT.get_paginator("list_objects_v2")
    fsa_ids: List[str] = []
    pat = re.compile(r"(.+)_train\.csv$", re.IGNORECASE)  # capture FSA ID part
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=TRAINING_PREFIX):
        for obj in page.get("Contents", []):
            filename = os.path.basename(obj["Key"])
            match = pat.match(filename)
            if match:
                fsa_ids.append(match.group(1))  # just the FSA ID
    fsa_ids.sort()
    return fsa_ids
