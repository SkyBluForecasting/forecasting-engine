import pandas as pd

# Mapping MLflow quantile names → DB columns
QUANTILE_MAP = {
    "quantile_p05": "p05",
    "quantile_p10": "p10",
    "quantile_p30": "p30",
    "quantile_p50": "p50",
    "quantile_p70": "p70",
    "quantile_p90": "p90",
    "quantile_p95": "p95",
}

# Possible timestamp columns
TIMESTAMP_ALIASES = ["timestamp", "ds", "time"]


class ForecastDataProcessor:
    """
    Handles preprocessing and preparation of measurement data
    before running a forecast.
    """

    def __init__(self, df: pd.DataFrame, pj):
        """
        Initialize with a measurements dataframe and prediction job metadata.
        """
        self.df = df.copy()
        self.pj = pj

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess dataframe by ensuring datetime index,
        sorting, and deduplicating.
        """
        df = self.df

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            else:
                raise ValueError(
                    "Input dataframe does not have a datetime index or 'timestamp' column."
                )

        # Sort and deduplicate
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()

        return df

    def add_forecast_horizon_nans(self) -> pd.DataFrame:
        """
        Preprocess historical data and append future NaN rows for the forecast horizon.
        Future horizon always starts at CURRENT UTC time.
        """
        # 1. Preprocess historical data
        df = self.preprocess().copy()

        # 2. Anchor horizon at current UTC time
        now = pd.Timestamp.utcnow().floor(f"{self.pj.resolution_minutes}min")
        horizon_steps = int(self.pj.horizon_minutes / self.pj.resolution_minutes)
        freq = f"{self.pj.resolution_minutes}min"

        # 3. Filter out any historical rows that are >= now
        df = df[df.index < now]

        # 4. Create future horizon DataFrame
        future_times = pd.date_range(start=now, periods=horizon_steps, freq=freq)
        horizon_df = pd.DataFrame(index=future_times)
        if "load" in df.columns:
            horizon_df["load"] = None

        # 5. Combine historical + horizon
        combined = pd.concat([df, horizon_df], ignore_index=False)

        # 6. Safety: remove duplicates if any (keeps horizon values)
        combined = combined[~combined.index.duplicated(keep="last")]

        return combined


def normalize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize forecast DataFrame columns:
      - Lowercase & strip spaces
      - Map MLflow quantile names to DB column names
      - Ensure 'timestamp' column exists (from index if needed)
      - Ensure 'forecast' column exists (fill with p50 if missing)
    """
    # Lowercase & strip
    df = df.rename(columns=lambda c: c.lower().strip())

    # Map quantiles
    df = df.rename(columns=QUANTILE_MAP)

    # If index is datetime, move it to 'timestamp'
    if isinstance(df.index, pd.DatetimeIndex) and "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})

    # Find timestamp column
    ts_col = next((c for c in TIMESTAMP_ALIASES if c in df.columns), None)
    if not ts_col:
        raise ValueError(
            f"Forecast DataFrame missing timestamp column. Expected one of {TIMESTAMP_ALIASES}"
        )

    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})

    # Convert timestamp to UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Ensure forecast column exists
    if "forecast" not in df.columns:
        if "p50" in df.columns:
            df["forecast"] = df["p50"]
        else:
            raise ValueError("Forecast DataFrame missing 'forecast' and 'p50' columns")

    return df
