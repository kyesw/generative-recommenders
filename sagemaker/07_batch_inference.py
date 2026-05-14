"""
Run batch inference using SageMaker Batch Transform.

Reuses the same inference container (inference.py + sagemaker_handler.py)
as the real-time endpoint. SageMaker starts the container, waits for /ping,
then streams each line of the input JSONL file to POST /invocations and
collects the responses into an output file on S3.

Input format (JSONL — one JSON object per line):
  {"user_id": 1, "sequence": [101, 202, 303], "top_k": 20}
  {"user_id": 2, "sequence": [404, 505], "top_k": 10}

  When "sequence" is provided, Feature Store is bypassed.
  When only "user_id" is provided, the container fetches the user's
  interaction history from Feature Store at request time.

Output format (JSONL — one JSON object per line):
  {"user_id": 1, "item_ids": [55, 78, ...]}
  {"user_id": 2, "item_ids": [12, 34, ...]}

Preparing input data:
  Write a JSONL file locally and upload to S3:
    aws s3 cp requests.jsonl s3://my-bucket/batch-input/requests.jsonl

Usage:
  python sagemaker/07_batch_inference.py \\
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \\
      --bucket my-bucket \\
      --region us-east-1 \\
      --mlflow-tracking-uri https://<tracking-server>.sagemaker.us-east-1.amazonaws.com \\
      --mlflow-run-id <run-id> \\
      --input-s3-uri s3://my-bucket/batch-input/requests.jsonl

Optional flags:
  --output-s3-uri       S3 prefix for output (default: s3://<bucket>/generative-recommenders/batch-output/<timestamp>/)
  --instance-type       Instance type (default: ml.g4dn.xlarge)
  --instance-count      Number of instances for parallel processing (default: 1)
  --max-concurrent      Max concurrent /invocations calls per instance (default: 1)
  --max-payload-mb      Max payload size in MB per request (default: 1)
  --job-name            Custom job name (auto-generated if omitted)
  --wait / --no-wait    Wait for the job to complete (default: --wait)
"""

import argparse
import logging
import os
import sys
import time

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference via SageMaker Batch Transform."
    )
    parser.add_argument("--role", required=True, help="IAM role ARN.")
    parser.add_argument("--bucket", required=True, help="S3 bucket for output.")
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    parser.add_argument(
        "--account",
        default=None,
        help="AWS account ID. Resolved via STS if not provided.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        required=True,
        dest="mlflow_tracking_uri",
        help="URI of the MLflow tracking server.",
    )
    parser.add_argument(
        "--mlflow-run-id",
        required=True,
        dest="mlflow_run_id",
        help="MLflow run ID to load model config and checkpoint from.",
    )
    parser.add_argument(
        "--input-s3-uri",
        required=True,
        dest="input_s3_uri",
        help="S3 URI of the input JSONL file (e.g. s3://bucket/input/requests.jsonl).",
    )
    parser.add_argument(
        "--output-s3-uri",
        default=None,
        dest="output_s3_uri",
        help="S3 prefix for output. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g4dn.xlarge",
        dest="instance_type",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        dest="instance_count",
        help="Number of instances (input file is sharded across them).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        dest="max_concurrent",
        help="Max concurrent /invocations calls per instance.",
    )
    parser.add_argument(
        "--max-payload-mb",
        type=int,
        default=1,
        dest="max_payload_mb",
        help="Max payload size in MB per request.",
    )
    parser.add_argument("--image-tag", default="latest", dest="image_tag")
    parser.add_argument(
        "--job-name",
        default=None,
        dest="job_name",
        help="Batch Transform job name. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--feature-store-region",
        default=None,
        dest="feature_store_region",
        help="AWS region for Feature Store (defaults to --region).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        dest="wait",
        help="Wait for the job to complete (default).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_false",
        dest="wait",
        help="Submit the job and return immediately.",
    )
    return parser.parse_args()


def get_account_id(account: str | None, region: str) -> str:
    if account:
        return account
    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def main() -> None:
    args = parse_args()

    import sagemaker
    from sagemaker.model import Model

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    account = get_account_id(args.account, args.region)
    image_uri = (
        f"{account}.dkr.ecr.{args.region}.amazonaws.com/"
        f"generative-recommenders:{args.image_tag}"
    )

    timestamp = int(time.time())
    job_name = args.job_name or f"gr-batch-{timestamp}"
    output_s3_uri = args.output_s3_uri or (
        f"s3://{args.bucket}/generative-recommenders/batch-output/{timestamp}/"
    )
    feature_store_region = args.feature_store_region or args.region

    env = {
        "SAGEMAKER_PROGRAM": "inference.py",
        "MLFLOW_TRACKING_URI": args.mlflow_tracking_uri,
        "MLFLOW_RUN_ID": args.mlflow_run_id,
        "FEATURE_STORE_REGION": feature_store_region,
    }

    logger.info(f"Image URI       : {image_uri}")
    logger.info(f"MLflow run ID   : {args.mlflow_run_id}")
    logger.info(f"Job name        : {job_name}")
    logger.info(f"Instance type   : {args.instance_type}")
    logger.info(f"Instance count  : {args.instance_count}")
    logger.info(f"Max concurrent  : {args.max_concurrent}")
    logger.info(f"Input           : {args.input_s3_uri}")
    logger.info(f"Output          : {output_s3_uri}")

    # -----------------------------------------------------------------
    # 1. Create SageMaker Model (same as deploy_endpoint.py)
    # -----------------------------------------------------------------
    model = Model(
        image_uri=image_uri,
        model_data=None,
        role=args.role,
        env=env,
        sagemaker_session=sm_session,
        name=f"gr-batch-model-{timestamp}",
    )

    # -----------------------------------------------------------------
    # 2. Create Transformer and launch Batch Transform job
    #
    #    SplitType='Line'          — split JSONL input by newline
    #    BatchStrategy='SingleRecord' — one record per /invocations call
    #    AssembleWith='Line'        — join output records with newline
    # -----------------------------------------------------------------
    transformer = model.transformer(
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=output_s3_uri,
        strategy="SingleRecord",
        assemble_with="Line",
        max_concurrent_transforms=args.max_concurrent,
        max_payload=args.max_payload_mb,
    )

    transformer.transform(
        data=args.input_s3_uri,
        content_type="application/json",
        split_type="Line",
        job_name=job_name,
        wait=args.wait,
        logs=args.wait,
    )

    if args.wait:
        logger.info(f"\nBatch Transform job '{job_name}' completed.")
        logger.info(f"Output written to: {output_s3_uri}")
        logger.info(
            f"\nDownload results:\n"
            f"  aws s3 cp --recursive {output_s3_uri} ./batch-output/"
        )
    else:
        logger.info(f"\nBatch Transform job '{job_name}' submitted.")
        logger.info(
            f"\nMonitor progress:\n"
            f"  aws sagemaker describe-transform-job --transform-job-name {job_name} --region {args.region}"
        )
        logger.info(f"Output will be written to: {output_s3_uri}")


if __name__ == "__main__":
    main()
