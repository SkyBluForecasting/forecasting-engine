# pragma: exclude file

"""
This is just a hacky script to train the L9M model for testing.
This will be replaced with a proper training script.
"""

from forecasting_engine.openstef.pipeline.train_model import train_model_pipeline

from forecasting_engine.openstef.data_classes.prediction_job import (
    PredictionJobDataClass,
)

from forecasting_engine.config import MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT

import pandas as pd
import os


def train_model():

    # Specify the FSA value you want to train a model with.
    fsa_id = "L9M"

    # Location of file for training
    train_data_path = os.path.join(
        os.path.dirname(__file__),
        "L9M_train.csv",
    )
    train_data = pd.read_csv(
        train_data_path,
        index_col="index",  # set the index column
        parse_dates=["index"],  # parse index column as datetime
    )

    # Define properties of training/prediction - a 'prediction_job'
    # This pj will generate forecasts at 15min increments at horizons 0.25h, 0.5h etc up
    # to 47h.
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

    train, val, test = train_model_pipeline(
        pj,
        train_data,
        check_old_model_age=False,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        artifact_folder=MLFLOW_ARTIFACT_ROOT,
    )


def main():
    train_model()


if __name__ == "__main__":
    main()
