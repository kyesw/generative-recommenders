"""
One-time setup: creates SageMaker Feature Store feature groups for
user interaction sequences and item metadata.

Both feature groups are created with online + offline store enabled so
the same groups can later serve real-time inference without recreation.

Usage:
  python sagemaker/feature_store_setup.py \\
      --bucket my-bucket \\
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \\
      --dataset-name ml-1m \\
      --region us-east-1

Feature groups created:
  user-interactions-{dataset_name}
      user_id (Integral, record identifier)
      sequence_item_ids (String)   — comma-separated item IDs
      sequence_ratings  (String)   — comma-separated ratings
      sequence_timestamps (String) — comma-separated timestamps
      event_time (Fractional)      — required by Feature Store

  item-metadata-{dataset_name}
      movie_id (Integral, record identifier)
      title (String)
      genres (String)              — pipe-separated
      year (String)
      cleaned_title (String)
      event_time (Fractional)      — required by Feature Store
"""

import argparse
import logging
import sys
import time

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 5
_TIMEOUT_S = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SageMaker Feature Store feature groups."
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket for offline store.")
    parser.add_argument(
        "--role", required=True, help="IAM role ARN with Feature Store permissions."
    )
    parser.add_argument(
        "--dataset-name",
        default="ml-1m",
        dest="dataset_name",
        help="Dataset identifier, appended to feature group names.",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region.",
    )
    return parser.parse_args()


def _wait_until_created(feature_group, timeout: int = _TIMEOUT_S) -> None:
    """Poll until the feature group status is 'Created'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = feature_group.describe()["FeatureGroupStatus"]
        logger.info(f"  status: {status}")
        if status == "Created":
            return
        if status in ("CreateFailed", "DeleteFailed"):
            raise RuntimeError(
                f"Feature group '{feature_group.name}' entered status: {status}"
            )
        time.sleep(_POLL_INTERVAL_S)
    raise TimeoutError(
        f"Feature group '{feature_group.name}' not ready after {timeout}s"
    )


def _create_feature_group(
    name: str,
    sample_df,
    record_identifier: str,
    s3_uri: str,
    role: str,
    session,
) -> None:
    """Create a feature group, skip if it already exists."""
    import pandas as pd
    from sagemaker.feature_store.feature_group import FeatureGroup

    fg = FeatureGroup(name=name, sagemaker_session=session)

    # Check existence first to make this script idempotent.
    try:
        status = fg.describe()["FeatureGroupStatus"]
        logger.info(f"Feature group '{name}' already exists (status: {status}), skipping.")
        return
    except Exception:
        pass  # Does not exist yet — proceed to create.

    fg.load_feature_definitions(data_frame=sample_df)
    fg.create(
        s3_uri=s3_uri,
        record_identifier_name=record_identifier,
        event_time_feature_name="event_time",
        role_arn=role,
        enable_online_store=True,
        # online_store_config={
        #    "StorageType": "InMemoryStorage"  # Redis
        # },
        # VPC Endpoint for SageMaker Feature Store — com.amazonaws.<region>.sagemaker.featurestore-runtime must exist in
        #   the VPC, otherwise the endpoint container can't reach Feature Store even within the VPC
        # online_store_config={
        #     "SecurityConfig": {
        #         "VpcConfig": {
        #             "SubnetIds": ["subnet-xxxxxxxx"],
        #             "SecurityGroupIds": ["sg-xxxxxxxx"],
        #         }
        #     }
        # },
    )
    
    logger.info(f"Creating feature group '{name}' — waiting for it to become active...")
    _wait_until_created(fg)
    logger.info(f"Feature group '{name}' is ready.")


def create_user_interactions_fg(
    session, bucket: str, role: str, dataset_name: str
) -> None:
    import pandas as pd

    name = f"user-interactions-{dataset_name}"
    s3_uri = f"s3://{bucket}/feature-store/{dataset_name}/"

    # Sample DataFrame — used only to infer feature types.
    # int64   → Integral
    # object  → String
    # float64 → Fractional
    sample = pd.DataFrame(
        {
            "user_id": pd.array([0], dtype="int64"),
            "sequence_item_ids": pd.array([""], dtype="object"),
            "sequence_ratings": pd.array([""], dtype="object"),
            "sequence_timestamps": pd.array([""], dtype="object"),
            "event_time": pd.array([time.time()], dtype="float64"),
        }
    )

    _create_feature_group(
        name=name,
        sample_df=sample,
        record_identifier="user_id",
        s3_uri=s3_uri,
        role=role,
        session=session,
    )


def create_item_metadata_fg(
    session, bucket: str, role: str, dataset_name: str
) -> None:
    import pandas as pd

    name = f"item-metadata-{dataset_name}"
    s3_uri = f"s3://{bucket}/feature-store/{dataset_name}/"

    sample = pd.DataFrame(
        {
            "movie_id": pd.array([0], dtype="int64"),
            "title": pd.array([""], dtype="object"),
            "genres": pd.array([""], dtype="object"),
            "year": pd.array([""], dtype="object"),
            "cleaned_title": pd.array([""], dtype="object"),
            "event_time": pd.array([time.time()], dtype="float64"),
        }
    )

    _create_feature_group(
        name=name,
        sample_df=sample,
        record_identifier="movie_id",
        s3_uri=s3_uri,
        role=role,
        session=session,
    )


def main() -> None:
    args = parse_args()

    import sagemaker

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    logger.info(f"Dataset      : {args.dataset_name}")
    logger.info(f"Bucket       : {args.bucket}")
    logger.info(f"Offline store: s3://{args.bucket}/feature-store/{args.dataset_name}/")

    create_user_interactions_fg(sm_session, args.bucket, args.role, args.dataset_name)
    create_item_metadata_fg(sm_session, args.bucket, args.role, args.dataset_name)

    logger.info("All feature groups created successfully.")
    logger.info(
        f"\nNext step — ingest data:\n"
        f"  python sagemaker/ingest_features.py \\\n"
        f"      --bucket {args.bucket} \\\n"
        f"      --dataset-name {args.dataset_name} \\\n"
        f"      --region {args.region}"
    )


if __name__ == "__main__":
    main()
