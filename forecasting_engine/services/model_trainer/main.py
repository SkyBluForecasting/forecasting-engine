import argparse
from forecasting_engine.shared.model_trainer import TrainingManager
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.shared.logger_factory import get_logger

logger = get_logger(__name__)


def run_training(trainer: TrainingManager, asset_id: str | None = None):
    """Core logic separated from CLI."""
    if asset_id:
        trainer.train_asset(asset_id.upper())
    else:
        trainer.train_all_assets()


def main(
    trainer_cls=TrainingManager, session_cls=SessionLocal, logger_factory=get_logger
):  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id")
    args = parser.parse_args()

    session = session_cls()

    trainer = trainer_cls(session)
    print("hi")

    try:
        run_training(trainer, args.asset_id)
    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    main()
