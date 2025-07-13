# Forecasting Engine

The **Forecasting Engine** is a Python-based service for generating short-term energy forecasts. It builds on the open-source [OpenSTEF](https://github.com/OpenSTEF/openstef) forecasting library, and adds orchestration components that connect to AWS S3, MLFlow, and a PostgreSQL database (for storing ML artifacts):

- Forecasting logic: Core model training and forecasting powered by OpenSTEF.
- MLflow: Tracks and stores trained forecasting models and metadata in a PostgreSQL-backed MLflow server. Artifacts (trained model files) are stored on AWS S3.
- AWS S3: Used as the central artifact store for models and forecast output data.
- PostgreSQL database: Stores MLflow metadata such as experiment and run info.
- Orchestration components: Custom code to trigger forecasts, retrieve data from S3, handle message queues (SQS), and coordinate pipeline execution.

This repo is designed to be deployed on an EC2 instance and serves as the backend forecasting engine in a larger forecasting system.

## What's in this repo

```
forecasting_engine/ **Included in deployments
├── openstef/ # Core licensed forecasting logic 
├── orchestration/ # Custom logic to run forecasts, load from S3, poll SQS. 
├── tasks/ # Executable scripts (polling, cron, CLI entrypoints) 
scripts/ # Test scripts for local testing ** NOT included in deployments
test/ # Unit tests  ** NOT included in deployments
```

## Prerequisites

- Python 3.11 or higher
- AWS credentials configured (for S3 access)
- pip (Python package manager)
- Docker (optional, for containerized deployment)


## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/mfavit/forecasting-engine.git
cd forecasting-engine
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r test-requirements.txt
```

4. If you want to run some of the code locally which integrates with S3 and the MLFLOW server - create a `.env` file in the root directory with your AWS IAM and MLFLOW configs. Replace the placeholders with actual values. 

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-2
S3_BUCKET=top-level-bucket-name (e.g., forecasting-forecasts)

# MLFLOW Config
MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI=http://<your-ec2-public-ip>:5050

# MLFLow for local dev
MLFLOW_DB_URI=get_from_supabase
MLFLOW_ARTIFACT_ROOT=s3://forecasting-forecasts/mlflow_trained_models/
```

## Starting the mlflow server locally:

```bash
mlflow server --backend-store-uri $MLFLOW_DB_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 127.0.0.1 --port 5050
```

Open your browser to `http://localhost:5050` to access MLFLow. 


## Running tests

```bash
pytest test
```

To run tests and also check coverage:

```bash
coverage run --source=forecasting_engine -m pytest test/orchestration/ && coverage report -m
```

## Running pre-commit



# About OpenSTEF

This project uses OpenSTEF, an open-source forecasting library maintained by Alliander and the Linux Foundation. OpenSTEF provides:

- Model training and forecasting pipelines
- Feature engineering logic
- Support for multiple forecast horizons and targets


# Table of contents

- [Table of contents](#table-of-contents)
- [External information sources](#external-information-sources)
- [Installation](#installation)
- [Usage](#usage)
  - [Example notebooks](#example-notebooks)
  - [Reference Implementation](#reference-implementation)
  - [Database connector for OpenSTEF](#database-connector-for-openstef)
- [License](license)
- [Contact](#contact)

# External information sources

- [Documentation website](https://openstef.github.io/openstef/index.html);
- [Python package](https://pypi.org/project/openstef/);
- [Linux Foundation project page](https://www.lfenergy.org/projects/openstef/);
- [Documentation on dashboard](https://raw.githack.com/OpenSTEF/.github/main/profile/html/openstef_dashboard_doc.html);
- [Video about OpenSTEF](https://www.lfenergy.org/forecasting-to-create-a-more-resilient-optimized-grid/);
Note: The OpenSTEF code is maintained in the openstef/ folder under the MPL-2.0 license.

# Usage

## Example notebooks

To help you get started, a set of fundamental example notebooks has been created. You can access these offline examples [here](https://github.com/OpenSTEF/openstef-offline-example).

## Reference Implementation

A complete implementation including databases, user interface, example data, etc. is available at: https://github.com/OpenSTEF/openstef-reference

![screenshot](https://user-images.githubusercontent.com/60883372/146760483-29af3ac7-62af-4f13-98c7-982a79c517d1.jpg)
Screenshot of the operational dashboard showing the key functionality of OpenSTEF.
Dashboard documentation can be found [here](https://raw.githack.com/OpenSTEF/.github/main/profile/html/openstef_dashboard_doc.html).

To run a task use:

```shell
python -m openstef task <task_name>
```

## Database connector for openstef

This repository provides an interface to OpenSTEF (reference) databases. The repository can be found [here](https://github.com/OpenSTEF/openstef-dbc).

# License

This project is licensed under the Mozilla Public License, version 2.0 - see LICENSE for details.

## Licenses third-party libraries

This project includes third-party libraries, which are licensed under their own respective Open-Source licenses. SPDX-License-Identifier headers are used to show which license is applicable. The concerning license files can be found in the LICENSES directory.

# Contributing

Please read [CODE_OF_CONDUCT.md](https://github.com/OpenSTEF/.github/blob/main/CODE_OF_CONDUCT.md), [CONTRIBUTING.md](https://github.com/OpenSTEF/.github/blob/main/CONTRIBUTING.md) and [PROJECT_GOVERNANCE.md](https://github.com/OpenSTEF/.github/blob/main/PROJECT_GOVERNANCE.md) for details on the process for submitting pull requests to us.

# Contact

Please read [SUPPORT.md](https://github.com/OpenSTEF/.github/blob/main/SUPPORT.md) for how to connect and get into contact with the OpenSTEF project
