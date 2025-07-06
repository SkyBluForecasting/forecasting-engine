<!--
SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

# Forecasting Engine

The **Forecasting Engine** is a Python-based service for generating short-term energy forecasts. It builds on the open-source [OpenSTEF](https://github.com/OpenSTEF/openstef) forecasting library, and adds orchestration components for:

- Automated forecasting via AWS SQS queue polling
- Integrating with S3 to pull forecasting input data and push generated forecasts 
- Scheduled training and forecast generation jobs

This repo is designed to be deployed on an EC2 instance and serves as the backend forecasting engine in a larger forecasting system.

## What's in this repo

forecasting-engine/ **Included in deployments
├── openstef/ # Core licensed forecasting logic 
├── orchestration/ # Custom logic to run forecasts, load from S3, poll SQS. 
├── jobs/ # Executable scripts (polling, cron, CLI entrypoints) 
scripts/ # Test scripts for local testing ** NOT included in deployments
test/ # Unit tests  ** NOT included in deployments


# Installation

## Install the forecasting-engine

```shell
pip install forecasting-engine
```

### Remark regarding installation within a **conda environment on Windows**

A version of the pywin32 package will be installed as a secondary dependency along with the installation of the openstef package. Since conda relies on an old version of pywin32, the new installation can break conda's functionality. The following command can solve this issue:

```shell
pip install pywin32==300
```

For more information on this issue see the [readme of pywin32](https://github.com/mhammond/pywin32#installing-via-pip) or [this Github issue](https://github.com/mhammond/pywin32/issues/1865#issue-1212752696).

## Remark regarding installation on Apple Silicon

If you want to install the `forecasting-engine` package on Apple Silicon (Mac with M1-chip or newer), you can encounter issues with the dependencies, such as `xgboost`. Solution:

1. Run `brew install libomp` (if you haven’t installed Homebrew: [follow instructions here](https://brew.sh/))
2. If your interpreter can not find the `libomp` installation in `/usr/local/bin`, it is probably in `/opt/brew/Cellar`. Run:

```sh
mkdir -p /usr/local/opt/libomp/
ln -s /opt/brew/Cellar/libomp/{your_version}/lib /usr/local/opt/libomp/lib
```

3. Uninstall `xgboost` with `pip` (`pip uninstall xgboost`) and install with `conda-forge` (`conda install -c conda-forge xgboost`)
4. If you encounter similar issues with `lightgbm`: uninstall `lightgbm` with `pip` (`pip uninstall lightgbm`) and install later version with `conda-forge` (`conda install -c conda-forge 'lightgbm>=4.2.0'`)

### Remark regarding installation with minimal XGBoost dependency

It is possible to install forecasting-engine with a minimal XGBoost (CPU-only) package. This only works on x86_64 (amd64) Linux and Windows platforms. Advantage is that significantly smaller dependencies are installed. In that case run:

```shell
pip install forecasting-engine[cpu]
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
