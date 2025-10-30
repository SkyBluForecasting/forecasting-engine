# Forecasting Engine

The **Forecasting Engine** is a Python-based service for generating short-term energy forecasts. It builds on the open-source [OpenSTEF](https://github.com/OpenSTEF/openstef) forecasting library, and has services that connect to an AWS SQS Queue, MLFlow, and a PostgreSQL database.

## What's in this repo

```
forecasting_engine/  # Included in deployments
├── openstef/         # Core forecasting library
├── db_io/            # Database access layer (sessions, reads/writes)
├── services/         # Long-running services (Dockerized)
│   ├── forecast_enqueuer/  # Checks DB, enqueues SQS forecast requests
│   ├── forecast_poller/    # Polls SQS and runs forecasts
│   └── model_trainer/      # Trains models with MLflow + S3
├── shared/           # Reusable business logic and utilities
│   ├── forecast_runner.py  # Core forecast generation
│   ├── forecast_utils.py   # Forecast data helpers
│   ├── logger_factory.py   # Logging setup
│   └── s3_utils.py         # S3 utilities
├── tasks/            # CLI entrypoints
scripts/              # Local testing helpers (not deployed)
test/                 # Unit/integration tests (not deployed)
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
./install_requirements.sh
```

4. Install the package in editable mode (so imports work if running things locally)

```bash
pip install -e .
```

5. If you want to run some of the code locally which integrates with a DB, an SQS queue, and the MLFLOW server - create a `.env` file in the root directory with your configs. Replace the placeholders with actual values. 

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-2
S3_BUCKET=top-level-bucket-name (e.g., forecasting-forecasts)

# MLFLOW Config
MLFLOW_TRACKING_URI=http://127.0.0.1:5050  # Localhost
MLFLOW_DB_URI=get_from_supabase_session_pooler
MLFLOW_ARTIFACT_ROOT=path-to-mlflow-s3-artifacts-folder

# Database Configuration
DATABASE_URL="postgresql://postgres:localpass@localhost:5432/appdb"
```

You may need to run `source .env` in order to set the variables in your virtual environment.

### Starting the mlflow server locally:

```bash
mlflow server --backend-store-uri $MLFLOW_DB_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 127.0.0.1 --port 5050
```

Open your browser to `http://localhost:5050` to access MLFLow. 

## Local Docker Setup

This project includes Docker configurations for containerized deployment and development. 

1. Generate a CodeArtifact Auth token so that internal packages can be downloaded.

```bash
export CODEARTIFACT_AUTH_TOKEN=$(aws codeartifact get-authorization-token \
  --domain skyblu \
  --domain-owner 591082451778 \
  --region $AWS_REGION \
  --query authorizationToken \
  --output text)
```

2. If you want to use a local DB, ensure it is running (see forecasting-db repo)

3. Build and start all services:
```bash
docker-compose -f docker-compose.dev.yml build --build-arg CODEARTIFACT_AUTH_TOKEN="$CODEARTIFACT_AUTH_TOKEN" \
                     --build-arg AWS_REGION="$AWS_REGION"

docker-compose -f docker-compose.dev.yml up -d
```

4. Stop all services:
```bash
docker-compose down
```

5. View logs:
```bash
docker-compose logs -f mlflow
docker-compose logs -f measurement-poller
```

### Accessing Services

- **MLflow UI**: http://localhost:5050
- **Forecasting container**: Execute commands inside the container. E.g, :
```bash
docker exec -it forecasting-tasks python forecasting_engine/tasks/run_single_forecast.py L9M
```

## Running tests

```bash
pytest test
```

To run tests and also check coverage:

```bash
coverage run --source=forecasting_engine -m pytest test/orchestration/ && coverage report -m
```

## Setting up pre-commit

It is recommended to enable your IDE to run the pre-commit checks before submitting a commit.

```bash
pip install pre-commit
pre-commit install
```

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
