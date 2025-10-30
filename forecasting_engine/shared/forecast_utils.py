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

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _compute_horizon_times(self) -> pd.DatetimeIndex:
        """Return DatetimeIndex for forecast horizon based on current time and job settings."""
        now = pd.Timestamp.utcnow().floor(f"{self.pj.resolution_minutes}min")
        steps = int(self.pj.horizon_minutes / self.pj.resolution_minutes)
        freq = f"{self.pj.resolution_minutes}min"
        return pd.date_range(start=now, periods=steps, freq=freq)

    @staticmethod
    def _clip_and_null_horizon(
        df: pd.DataFrame, horizon_times: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Clip df to end of horizon and null out 'load' for horizon timestamps."""
        df = df[df.index <= horizon_times[-1]].copy()
        if "load" not in df.columns:
            df["load"] = None
        df.loc[df.index.isin(horizon_times), "load"] = None
        return df

    @staticmethod
    def _fill_missing_horizon_rows(
        df: pd.DataFrame, horizon_times: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Ensure all horizon timestamps exist in df with load=None if missing."""
        missing = [t for t in horizon_times if t not in df.index]
        if not missing:
            return df

        horizon_df = pd.DataFrame(index=missing, columns=df.columns)
        horizon_df["load"] = None
        df = pd.concat([df, horizon_df])
        df = df.sort_index()
        return df

    # ---------------------------------------------------------
    # Main function
    # ---------------------------------------------------------

    def add_forecast_horizon_nans(self) -> pd.DataFrame:
        """Append forecast horizon timestamps with load=None and ensure complete coverage."""
        df = self.preprocess().copy()
        df.index = pd.to_datetime(df.index)

        horizon_times = self._compute_horizon_times()
        df = self._clip_and_null_horizon(df, horizon_times)
        df = self._fill_missing_horizon_rows(df, horizon_times)

        return df


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
