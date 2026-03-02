"""
Deploy a trained HSTU model as a SageMaker real-time inference endpoint.

The inference container fetches all hyperparameters and the model checkpoint
directly from MLflow, so no model-data S3 path is required.

Usage:
  python sagemaker/deploy_endpoint.py \\
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \\
      --region ap-northeast-2 \\
      --mlflow-tracking-uri https://<tracking-server-id>.sagemaker.ap-northeast-2.amazonaws.com \\
      --mlflow-run-id <run-id>

Optional flags:
  --endpoint-name         Name for the endpoint (default: generative-recommenders-<timestamp>)
  --instance-type         Instance type (default: ml.c5.xlarge)
  --image-tag             ECR image tag (default: latest)
  --account               AWS account ID (resolved via STS if omitted)
  --feature-store-region  AWS region for Feature Store (default: same as --region)
  --cache-size            Max users to cache (default: 10000)
  --cache-ttl             Cache TTL in seconds (default: 300)

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
import os
import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy HSTU model to a SageMaker real-time endpoint."
    )
    parser.add_argument("--role", required=True, help="IAM role ARN.")
    parser.add_argument("--region", default="us-east-1")
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
        help="MLflow run ID to load hyperparameters and checkpoint from.",
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
    parser.add_argument(
        "--feature-store-region",
        default=None,
        dest="feature_store_region",
        help="AWS region for Feature Store (defaults to --region).",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        dest="cache_size",
        help="Max users to cache (default: 10000, 0 to disable).",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=300,
        dest="cache_ttl",
        help="Cache TTL in seconds (default: 300).",
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
    from sagemaker.network import NetworkConfig 

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    account = get_account_id(args.account, args.region)
    image_uri = (
        f"{account}.dkr.ecr.{args.region}.amazonaws.com/"
        f"generative-recommenders:{args.image_tag}"
    )

    endpoint_name = args.endpoint_name or (
        f"generative-recommenders-{int(time.time())}"
    )
    feature_store_region = args.feature_store_region or args.region

    env = {
        "SAGEMAKER_PROGRAM": "inference.py",
        "MLFLOW_TRACKING_URI": args.mlflow_tracking_uri,
        "MLFLOW_RUN_ID": args.mlflow_run_id,
        "FEATURE_STORE_REGION": feature_store_region,
        "FEATURE_CACHE_SIZE": str(args.cache_size),
        "FEATURE_CACHE_TTL": str(args.cache_ttl),
    }

    logger.info(f"Image URI     : {image_uri}")
    logger.info(f"MLflow run ID : {args.mlflow_run_id}")
    logger.info(f"Endpoint name : {endpoint_name}")
    logger.info(f"Instance type : {args.instance_type}")
    logger.info(f"Environment   : {env}")

    # network_config = NetworkConfig(
    #     subnets=["subnet-xxxxxxxx"],
    #     security_group_ids=["sg-xxxxxxxx"],
    # )
    
    # model_data is None — the container downloads weights from MLflow directly.
    model = Model(
        image_uri=image_uri,
        model_data=None,
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
        # network_config=network_config,
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
