"""
Deploy a trained HSTU model as a SageMaker real-time inference endpoint.

Usage:
  python sagemaker/deploy_endpoint.py \\
      --model-data s3://my-bucket/generative-recommenders/output/<job>/output/model.tar.gz \\
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \\
      --bucket my-bucket \\
      --region ap-northeast-2

  # Or look up the latest training job automatically:
  python sagemaker/deploy_endpoint.py \\
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \\
      --bucket my-bucket \\
      --region ap-northeast-2

Optional flags:
  --endpoint-name       Name for the endpoint (default: generative-recommenders-<timestamp>)
  --instance-type       Instance type (default: ml.c5.xlarge; GPU not required for ml-1m)
  --image-tag           ECR image tag (default: latest)
  --dataset-name        Dataset name passed to inference script (default: ml-1m)
  --gin-config-file     Gin config path relative to /opt/ml/code (default: ml-1m large)
  --feature-store-region  AWS region for Feature Store (default: same as --region)
  --mlflow-tracking-uri   Optional MLflow server URI for logging

Inference request format (POST to endpoint):
  {"user_id": 42, "top_k": 10}
  {"user_id": 42, "top_k": 10, "sequence": [1, 2, 3]}  # bypass Feature Store

Inference response format:
  {"user_id": 42, "item_ids": [101, 55, 203, ...]}
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
        description="Deploy HSTU model to a SageMaker real-time endpoint."
    )
    parser.add_argument(
        "--model-data",
        default=None,
        dest="model_data",
        help=(
            "S3 URI of model.tar.gz produced by a training job. "
            "If omitted, the latest completed training job is used."
        ),
    )
    parser.add_argument("--role", required=True, help="IAM role ARN.")
    parser.add_argument("--bucket", required=True, help="S3 bucket (for ECR account resolution).")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--account",
        default=None,
        help="AWS account ID. Resolved via STS if not provided.",
    )
    parser.add_argument(
        "--endpoint-name",
        default=None,
        dest="endpoint_name",
        help="SageMaker endpoint name. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.c5.xlarge",
        dest="instance_type",
    )
    parser.add_argument("--image-tag", default="latest", dest="image_tag")
    parser.add_argument("--dataset-name", default="ml-1m", dest="dataset_name")
    parser.add_argument(
        "--gin-config-file",
        default="configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin",
        dest="gin_config_file",
    )
    parser.add_argument(
        "--feature-store-region",
        default=None,
        dest="feature_store_region",
        help="AWS region for Feature Store (defaults to --region).",
    )
    return parser.parse_args()


def get_account_id(account: str | None, region: str) -> str:
    if account:
        return account
    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def get_latest_model_data(bucket: str, region: str) -> str:
    """Find the most recent model.tar.gz under the default output prefix."""
    s3 = boto3.client("s3", region_name=region)
    prefix = "generative-recommenders/output/"
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    candidates = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("model.tar.gz"):
                candidates.append((obj["LastModified"], obj["Key"]))

    if not candidates:
        raise FileNotFoundError(
            f"No model.tar.gz found under s3://{bucket}/{prefix}"
        )
    candidates.sort(reverse=True)
    key = candidates[0][1]
    uri = f"s3://{bucket}/{key}"
    logger.info(f"Using latest model artifact: {uri}")
    return uri


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

    model_data = args.model_data or get_latest_model_data(args.bucket, args.region)

    endpoint_name = args.endpoint_name or (
        f"generative-recommenders-{int(time.time())}"
    )
    feature_store_region = args.feature_store_region or args.region

    env = {
        "SAGEMAKER_PROGRAM": "inference.py",
        "GIN_CONFIG_FILE": args.gin_config_file,
        "DATASET_NAME": args.dataset_name,
        "FEATURE_STORE_REGION": feature_store_region,
    }

    logger.info(f"Image URI     : {image_uri}")
    logger.info(f"Model data    : {model_data}")
    logger.info(f"Endpoint name : {endpoint_name}")
    logger.info(f"Instance type : {args.instance_type}")
    logger.info(f"Environment   : {env}")

    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=args.role,
        env=env,
        sagemaker_session=sm_session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=600,  # 10 minutes for model loading
        wait=True,
    )

    logger.info(f"\nEndpoint '{endpoint_name}' is live.")
    logger.info(
        f"\nTest it:\n"
        f"  python -c \"\n"
        f"import boto3, json\n"
        f"sm = boto3.client('sagemaker-runtime', region_name='{args.region}')\n"
        f"resp = sm.invoke_endpoint(\n"
        f"    EndpointName='{endpoint_name}',\n"
        f"    ContentType='application/json',\n"
        f"    Body=json.dumps({{'user_id': 1, 'top_k': 10}}),\n"
        f")\n"
        f"print(json.loads(resp['Body'].read()))\n"
        f"\""
    )


if __name__ == "__main__":
    main()
