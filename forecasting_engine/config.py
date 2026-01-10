import os
from dotenv import load_dotenv
import boto3

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
QUEUE_URL = os.getenv("SQS_QUEUE_URL")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT")
DATABASE_URL = os.getenv("DATABASE_URL")

# Create boto3 clients here, reuse everywhere
SQS_CLIENT = boto3.client("sqs", region_name=AWS_REGION)

DEFAULT_FORECAST_HORIZON_MINUTES = 48 * 60  # 48 hours
