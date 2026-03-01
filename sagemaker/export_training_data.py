"""
Export training data from the Feature Store offline store to S3 via Athena.

Queries the offline store (Glue/Athena), deduplicates to the latest record
per entity, and writes CSVs to an S3 location compatible with the training
pipeline directory structure.

Usage:
  python sagemaker/export_training_data.py \\
      --bucket my-bucket \\
      --dataset-name ml-1m \\
      --region us-east-1

Output S3 structure (pass the base URI to launch_training.py --data-s3-uri):
  s3://{bucket}/feature-store-export/{dataset_name}/
      {dataset_name}/sasrec_format.csv       ← user interaction sequences
      processed/{dataset_name}/movies.csv    ← item metadata

The deduplication query keeps the latest record per entity based on
api_invocation_time, so re-running ingest + export always produces
a fresh snapshot.

Athena query results are written to:
  s3://{bucket}/feature-store-export/{dataset_name}/athena-tmp/
and cleaned up after each export.
"""

import argparse
import logging
import sys
import time

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

_ATHENA_DB = "sagemaker_featurestore"
_POLL_INTERVAL_S = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Feature Store offline data to S3 for training."
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket.")
    parser.add_argument(
        "--dataset-name",
        default="ml-1m",
        dest="dataset_name",
        help="Dataset identifier matching the feature group names.",
    )
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def _latest_records_query(table: str, partition_key: str, select_cols: str) -> str:
    """Return an Athena SQL query that deduplicates to the latest record per entity."""
    return f"""
SELECT {select_cols}
FROM (
    SELECT
        {select_cols},
        ROW_NUMBER() OVER (
            PARTITION BY {partition_key}
            ORDER BY api_invocation_time DESC, write_time DESC
        ) AS rn
    FROM "{_ATHENA_DB}"."{table}"
    WHERE NOT is_deleted
) t
WHERE rn = 1
""".strip()


def _run_athena_query(fg, query_string: str, output_s3_uri: str):
    """Run an Athena query via the feature group helper and return a DataFrame."""
    athena_query = fg.athena_query()
    logger.info(f"  Athena table: {athena_query.table_name}")
    logger.info(f"  Query:\n{query_string}")
    athena_query.run(
        query_string=query_string,
        output_location=output_s3_uri,
    )
    athena_query.wait()
    df = athena_query.as_dataframe()
    logger.info(f"  Query returned {len(df)} rows.")
    return df


def _upload_csv(df, bucket: str, s3_key: str, s3_client) -> str:
    """Write DataFrame as CSV and upload to S3. Returns the full S3 URI."""
    import io
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=csv_buffer.getvalue().encode("utf-8"),
    )
    uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"  Uploaded to: {uri}")
    return uri


def export_user_interactions(
    dataset_name: str,
    bucket: str,
    athena_tmp_uri: str,
    session,
    s3_client,
) -> None:
    from sagemaker.feature_store.feature_group import FeatureGroup

    fg_name = f"user-interactions-{dataset_name}"
    logger.info(f"Exporting '{fg_name}'...")

    fg = FeatureGroup(name=fg_name, sagemaker_session=session)
    athena_query = fg.athena_query()
    table = athena_query.table_name

    query = _latest_records_query(
        table=table,
        partition_key="user_id",
        select_cols=(
            "CAST(user_id AS VARCHAR) AS user_id, "
            "sequence_item_ids, sequence_ratings, sequence_timestamps"
        ),
    )

    df = _run_athena_query(fg, query, athena_tmp_uri)

    # Rename CAST result back to user_id (already named correctly).
    # Cast user_id to int for compatibility with DatasetV2.
    df["user_id"] = df["user_id"].astype(int)

    s3_key = f"feature-store-export/{dataset_name}/{dataset_name}/sasrec_format.csv"
    _upload_csv(df, bucket, s3_key, s3_client)


def export_item_metadata(
    dataset_name: str,
    bucket: str,
    athena_tmp_uri: str,
    session,
    s3_client,
) -> None:
    from sagemaker.feature_store.feature_group import FeatureGroup

    fg_name = f"item-metadata-{dataset_name}"
    logger.info(f"Exporting '{fg_name}'...")

    fg = FeatureGroup(name=fg_name, sagemaker_session=session)
    athena_query = fg.athena_query()
    table = athena_query.table_name

    query = _latest_records_query(
        table=table,
        partition_key="movie_id",
        select_cols="movie_id, title, genres, year, cleaned_title",
    )

    df = _run_athena_query(fg, query, athena_tmp_uri)
    df["movie_id"] = df["movie_id"].astype(int)

    s3_key = f"feature-store-export/{dataset_name}/processed/{dataset_name}/movies.csv"
    _upload_csv(df, bucket, s3_key, s3_client)


def main() -> None:
    args = parse_args()

    import sagemaker

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)
    s3_client = boto_session.client("s3")

    athena_tmp_uri = (
        f"s3://{args.bucket}/feature-store-export/{args.dataset_name}/athena-tmp/"
    )
    output_base = f"s3://{args.bucket}/feature-store-export/{args.dataset_name}/"

    logger.info(f"Dataset       : {args.dataset_name}")
    logger.info(f"Output base   : {output_base}")
    logger.info(f"Athena tmp    : {athena_tmp_uri}")

    export_user_interactions(
        dataset_name=args.dataset_name,
        bucket=args.bucket,
        athena_tmp_uri=athena_tmp_uri,
        session=sm_session,
        s3_client=s3_client,
    )
    export_item_metadata(
        dataset_name=args.dataset_name,
        bucket=args.bucket,
        athena_tmp_uri=athena_tmp_uri,
        session=sm_session,
        s3_client=s3_client,
    )

    logger.info("\nExport complete.")
    logger.info(
        f"\nNext step — launch training:\n"
        f"  python sagemaker/launch_training.py \\\n"
        f"      --pipeline research \\\n"
        f"      --role <ROLE_ARN> \\\n"
        f"      --bucket {args.bucket} \\\n"
        f"      --data-s3-uri {output_base}"
    )


if __name__ == "__main__":
    main()
