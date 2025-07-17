import structlog
import logging


def get_logger(name: str):
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(pad_event=30),
        ],
    )
    return structlog.get_logger(name)
