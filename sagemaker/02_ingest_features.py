"""
Ingest preprocessed data into SageMaker Feature Store.

Reads the CSVs produced by prepare_data.py and batch-ingests them into
the feature groups created by feature_store_setup.py.

Run this after feature_store_setup.py and after prepare_data.py has
produced the preprocessed CSV files.

Usage:
  python sagemaker/ingest_features.py \\
      --bucket my-bucket \\
      --dataset-name ml-1m \\
      --region us-east-1 \\
      --data-dir /tmp          # directory that contains {dataset_name}/ subtree

The script reads:
  {data_dir}/{dataset_name}/sasrec_format.csv
  {data_dir}/processed/{dataset_name}/movies.csv

And ingests into:
  user-interactions-{dataset_name}
  item-metadata-{dataset_name}
"""

import argparse
import logging
import sys
import time

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest preprocessed CSVs into SageMaker Feature Store."
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket (for session context).")
    parser.add_argument(
        "--dataset-name",
        default="ml-1m",
        dest="dataset_name",
        help="Dataset identifier matching the feature group names.",
    )
    parser.add_argument(
        "--data-dir",
        default="/tmp",
        dest="data_dir",
        help="Local root directory containing the preprocessed CSV subtree.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        dest="max_workers",
        help="Parallel workers for Feature Store ingestion.",
    )
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def ingest_user_interactions(
    dataset_name: str,
    data_dir: str,
    max_workers: int,
    session,
) -> None:
    import pandas as pd
    from sagemaker.feature_store.feature_group import FeatureGroup

    csv_path = f"{data_dir}/{dataset_name}/sasrec_format.csv"
    logger.info(f"Reading user interactions from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Keep only the columns the feature group expects.
    df = df[["user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"]].copy()
    df["user_id"] = df["user_id"].astype("int64")
    # event_time: current ingestion time (float Unix seconds required by Feature Store).
    df["event_time"] = float(time.time())

    fg_name = f"user-interactions-{dataset_name}"
    logger.info(f"Ingesting {len(df)} user records into '{fg_name}'...")
    fg = FeatureGroup(name=fg_name, sagemaker_session=session)
    fg.ingest(data_frame=df, max_workers=max_workers, wait=True)
    logger.info(f"Ingestion complete for '{fg_name}'.")


def ingest_item_metadata(
    dataset_name: str,
    data_dir: str,
    max_workers: int,
    session,
) -> None:
    import pandas as pd
    from sagemaker.feature_store.feature_group import FeatureGroup

    csv_path = f"{data_dir}/processed/{dataset_name}/movies.csv"
    logger.info(f"Reading item metadata from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Keep only the columns the feature group expects.
    df = df[["movie_id", "title", "genres", "year", "cleaned_title"]].copy()
    df["movie_id"] = df["movie_id"].astype("int64")
    # year may be parsed as int — cast to str to match String feature type.
    df["year"] = df["year"].astype(str)
    df["event_time"] = float(time.time())

    fg_name = f"item-metadata-{dataset_name}"
    logger.info(f"Ingesting {len(df)} item records into '{fg_name}'...")
    fg = FeatureGroup(name=fg_name, sagemaker_session=session)
    fg.ingest(data_frame=df, max_workers=max_workers, wait=True)
    logger.info(f"Ingestion complete for '{fg_name}'.")


def main() -> None:
    args = parse_args()

    import sagemaker

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    logger.info(f"Dataset  : {args.dataset_name}")
    logger.info(f"Data dir : {args.data_dir}")
    logger.info(f"Workers  : {args.max_workers}")

    ingest_user_interactions(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        max_workers=args.max_workers,
        session=sm_session,
    )
    ingest_item_metadata(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        max_workers=args.max_workers,
        session=sm_session,
    )

    logger.info("All data ingested into Feature Store.")
    logger.info(
        f"\nNext step — export training dataset:\n"
        f"  python sagemaker/export_training_data.py \\\n"
        f"      --bucket {args.bucket} \\\n"
        f"      --dataset-name {args.dataset_name} \\\n"
        f"      --region {args.region}"
    )


if __name__ == "__main__":
    main()
