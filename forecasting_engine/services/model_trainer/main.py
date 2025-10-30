import argparse
from typing import List, Dict, Any

from forecasting_engine.shared.logger_factory import get_logger

from forecasting_engine.shared.s3_utils import (
    list_training_fsa_ids,
)

from forecasting_engine.shared.model_trainer import train_single_fsa


logger = get_logger(__name__)


def train_all():
    keys = list_training_fsa_ids()
    results: List[Dict[str, Any]] = []
    for i, key in enumerate(keys, start=1):
        logger.info(f"[{i}/{len(keys)}] Training {key} ...", flush=True)
        results.append(train_single_fsa(key))
    return results


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(
        description="Train OpenSTEF models on EC2 using S3 training data + MLflow."
    )
    ap.add_argument(
        "--fsa-id",
        help="Train only this FSA (e.g., L9M). If omitted, trains all *_train.csv in the prefix.",
    )
    args = ap.parse_args()

    if args.fsa_id:
        train_single_fsa(args.fsa_id.upper())
    else:
        train_all()


if __name__ == "__main__":  # pragma: no cover
    main()
