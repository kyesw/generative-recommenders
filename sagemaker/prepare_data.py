"""
One-time script to prepare training data and upload it to S3.

Two modes:

  sample   Download a public MovieLens dataset, preprocess it, and upload
           the resulting sasrec_format.csv to S3.

  custom   Upload your own pre-formatted CSV file directly to S3.
           The file must have the sasrec_format columns:
             user_id, sequence_item_ids, sequence_ratings, sequence_timestamps

Usage — sample data:
  python sagemaker/prepare_data.py sample \\
      --bucket my-bucket \\
      --dataset-name ml-1m

Usage — custom data:
  python sagemaker/prepare_data.py custom \\
      --bucket my-bucket \\
      --local-path /path/to/my_data.csv \\
      --dataset-name my-dataset

In both cases the script prints the S3 URI to pass to launch_training.py
via --data-s3-uri.
"""

import argparse
import logging
import os
import sys

# Add the repo root to sys.path so generative_recommenders is importable
# without requiring a full `pip install -e .` (which needs fbgemm_gpu/torchrec).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

S3_PREFIX = "generative-recommenders/data"


def s3_uri(bucket: str, dataset_name: str) -> str:
    return f"s3://{bucket}/{S3_PREFIX}/{dataset_name}/"


def upload_dir(local_dir: str, bucket: str, dataset_name: str) -> str:
    """Upload every file under local_dir to S3, preserving relative paths."""
    s3_client = boto3.client("s3")
    uploaded = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, local_dir)
            s3_key = f"{S3_PREFIX}/{dataset_name}/{rel_path}"
            logger.info(f"Uploading {abs_path} -> s3://{bucket}/{s3_key}")
            s3_client.upload_file(abs_path, bucket, s3_key)
            uploaded += 1
    uri = s3_uri(bucket, dataset_name)
    logger.info(f"Uploaded {uploaded} file(s). S3 URI: {uri}")
    return uri


def prepare_sample(args: argparse.Namespace) -> None:
    """Download and preprocess a public MovieLens dataset, then upload to S3."""
    dataset_name: str = args.dataset_name

    # chdir to / so the preprocessor's relative "tmp/" paths resolve to /tmp/
    original_cwd = os.getcwd()
    os.chdir("/")

    try:
        from generative_recommenders.research.data.preprocessor import (
            get_common_preprocessors,
        )

        preprocessors = get_common_preprocessors()
        if dataset_name not in preprocessors:
            raise ValueError(
                f"Unknown sample dataset '{dataset_name}'. "
                f"Supported: {list(preprocessors.keys())}"
            )

        logger.info(f"Preprocessing {dataset_name} (this may take a few minutes)...")
        preprocessors[dataset_name].preprocess_rating()

        if not os.path.isdir("/tmp"):
            raise FileNotFoundError("Expected /tmp to contain preprocessed data.")

        # Upload the full /tmp/ tree so the container can symlink it as tmp/
        uri = upload_dir("/tmp", args.bucket, dataset_name)
    finally:
        os.chdir(original_cwd)

    print(f"\nS3 URI (pass to --data-s3-uri):\n  {uri}")


def prepare_custom(args: argparse.Namespace) -> None:
    """Upload a user-provided sasrec_format CSV to S3."""
    local_path: str = args.local_path
    dataset_name: str = args.dataset_name

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    uri = upload_file(local_path, args.bucket, dataset_name)
    print(f"\nS3 URI (pass to --data-s3-uri):\n  {uri}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and upload training data to S3."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- sample mode ---
    sample_parser = subparsers.add_parser(
        "sample",
        help="Download and preprocess a public MovieLens dataset.",
    )
    sample_parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket to upload data to.",
    )
    sample_parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="AWS region.",
    )
    sample_parser.add_argument(
        "--dataset-name",
        default="ml-1m",
        dest="dataset_name",
        choices=["ml-1m", "ml-20m", "ml-1b", "amzn-books"],
        help="Dataset to download and preprocess.",
    )

    # --- custom mode ---
    custom_parser = subparsers.add_parser(
        "custom",
        help="Upload your own pre-formatted sasrec_format CSV.",
    )
    custom_parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket to upload data to.",
    )
    custom_parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="AWS region.",
    )
    custom_parser.add_argument(
        "--local-path",
        required=True,
        dest="local_path",
        help="Local path to your sasrec_format CSV file.",
    )
    custom_parser.add_argument(
        "--dataset-name",
        required=True,
        dest="dataset_name",
        help="Name to identify this dataset (used as S3 key prefix and hyperparameter).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    boto3.setup_default_session(region_name=args.region)

    if args.mode == "sample":
        prepare_sample(args)
    elif args.mode == "custom":
        prepare_custom(args)


if __name__ == "__main__":
    main()
