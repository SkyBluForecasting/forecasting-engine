import sys

sys.path.append("/Users/mfavit/forecasting-engine/")

import pandas as pd
import webbrowser
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from openstef.pipeline.train_model import train_model_pipeline
from openstef.pipeline.create_forecast import create_forecast_pipeline
from openstef.data_classes.prediction_job import PredictionJobDataClass


import pandas as pd
from io import StringIO


def generate_three_day_hourly_index(start_datetime: datetime) -> pd.DatetimeIndex:
    end_datetime = start_datetime + timedelta(days=3)
    return pd.date_range(
        start=start_datetime, end=end_datetime, freq="H", tz=start_datetime.tzinfo
    )


def create_fsa_data(file_path: str, fsa_value: str):
    # Read the file, skipping lines starting with "\\"
    with open(file_path, "r") as file:
        lines = file.readlines()
    data_lines = [line for line in lines if not line.startswith("\\")]

    # Join only the data lines
    clean_data = "".join(data_lines)

    df = pd.read_csv(StringIO(clean_data))

    # Confirm columns
    if "FSA" not in df.columns:
        print(f"Columns found: {df.columns}")
        raise ValueError("Expected 'FSA' column not found after cleaning the file.")

    # Filter by FSA
    df_fsa = df[df["FSA"] == fsa_value]

    if df_fsa.empty:
        print(f"No data found for FSA: {fsa_value}")
        return

    # Create datetime index
    df_fsa["index"] = pd.to_datetime(df_fsa["DATE"]) + pd.to_timedelta(
        df_fsa["HOUR"] - 1, unit="h"
    )

    # Group by index and sum the TOTAL_CONSUMPTION
    # NOTE: Could play with this and have the type of customer be an input, to only include certain customer types / train with it.
    result = df_fsa.groupby("index", as_index=False)["TOTAL_CONSUMPTION"].sum()

    # Format index
    result["index"] = result["index"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    result = result.rename(columns={"TOTAL_CONSUMPTION": "load"})
    result["index"] = pd.to_datetime(result["index"])  # convert string back to datetime
    result.set_index("index", inplace=True)  # set as index here
    return result


# Specify the FSA value you want to train a model with.
fsa_id = "L9M"

# Location of file for training
training_filename = "raw_ieso_data/PUB_HourlyConsumptionByFSA_202412_v1.csv"

# Location of file for forecasting
forecasting_filename = "raw_ieso_data/PUB_HourlyConsumptionByFSA_202412_v1.csv"

## Transform the raw input file from IESO into a training data set. Save to CSV.
train_data = create_fsa_data(training_filename, fsa_id)
train_data.to_csv(f"data/{fsa_id}_train.csv", index=True)
print(f"CSV file 'data/{fsa_id}_train.csv' created successfully.")

# Define properties of training/prediction - a 'prediction_job'
# This pj will generate forecasts at 15min increments at horizons 0.25h, 0.5h etc up to 47h.
pj = dict(
    id=fsa_id,
    model="xgb",
    quantiles=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    forecast_type="demand",  # load forecast
    lat=52.0,
    lon=5.0,
    horizon_minutes=47 * 60,  # 47 hours
    resolution_minutes=15,  # 15 min steps
    name="Example",
    hyper_params={},
    feature_names=None,
    default_modelspecs=None,
    save_train_forecasts=True,
)
pj = PredictionJobDataClass(**pj)

# Train the model.
# The dataset gets split into a training, validation and test set.In this case, we're not backtesting, so there will be no test set.
# Training set --> used to train the model and minimize errors between forecasts and true load values in that window.
# Validation set --> make predictions on this set to tune the hyperparameters.
# Validation set will often include more outlier data
# Store the model and reports in ./mlflow_artifacts and ./mlflow_trained_models.
# There is one trained model folder for each prediction job. If you run a pipeline again, if the new model is better than the old model, it will be saved in the same folder.

train, val, test = train_model_pipeline(
    pj,
    train_data,
    check_old_model_age=False,
    mlflow_tracking_uri="./mlflow_trained_models",
    artifact_folder="./mlflow_artifacts",
)


# The predictor plots show the "in sample predictions" of the train, test and validation data. How the model "would have " forecasted those points.
# The predictor plot numbers are the forecast horizons. E.g., Predictor0.25.html plots 0.25h hours (15 min) ahead for every timestamp in the dataset.
# Basically, it's plotting the two extreme forecasts (15min ahead and 47h ahead).
# The weight plot shows the importance and weight of every feature.

html_path = os.path.abspath(f"./mlflow_artifacts/{fsa_id}/Predictor0.25.html")
webbrowser.open(f"file://{html_path}")
html_path = os.path.abspath(f"./mlflow_artifacts/{fsa_id}/weight_plot.html")
webbrowser.open(f"file://{html_path}")

# # Prepare data such that a forecast can be made using the trained model.
input_dataset = create_fsa_data(forecasting_filename, fsa_id)

# Split in training and forecasting data
train_data = input_dataset.iloc[:-48, :]  # everything except last 48 rows (~ 48 hours)
test_indices = input_dataset.iloc[-48:, :].index  # last 48 rows

actual_load = input_dataset.loc[test_indices, "load"].copy(deep=True)

forecasted_load = input_dataset.copy(deep=True)
forecasted_load.loc[test_indices, "load"] = np.nan

forecast = create_forecast_pipeline(
    pj, forecasted_load, mlflow_tracking_uri="./mlflow_trained_models"
)
forecast["load"] = actual_load

forecast[["forecast", "load"]].plot()
plt.show()
