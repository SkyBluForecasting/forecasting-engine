import argparse

from forecasting_engine.shared.logger_factory import get_logger
from forecasting_engine.db_io.session import SessionLocal
from forecasting_engine.shared.model_trainer import TrainingManager


logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description="Train OpenSTEF models using measurements in the DB"
    )
    ap.add_argument(
        "--asset-id",
        help="Train only this asset. If omitted, trains all assets in the DB.",
    )
    args = ap.parse_args()

    # create a DB session
    session = SessionLocal()
    trainer = TrainingManager(session)

    if args.asset_id:
        logger.info(f"Starting training for single asset: {args.asset_id}")
        trainer.train_asset(args.asset_id.upper())
    else:
        logger.info("Starting training for all assets in DB...")
        trainer.train_all_assets()

    session.close()


if __name__ == "__main__":  # pragma: no cover
    main()
