# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from forecasting_engine.openstef.logging.logger_types import LoggerType
from forecasting_engine.openstef.logging.standard_logger import StandardLogger
from forecasting_engine.openstef.logging.structlog_logger import StructlogLogger
from forecasting_engine.openstef.settings import Settings


def get_logger(name: str, logger_type: str = Settings.logger_type):
    if logger_type == LoggerType.STANDARD:
        return StandardLogger(name)
    elif logger_type == LoggerType.STRUCTLOG:
        return StructlogLogger(name)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
