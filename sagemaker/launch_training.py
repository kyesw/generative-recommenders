"""
Submit a SageMaker training job for the generative-recommenders project.

Usage:
  python sagemaker/launch_training.py \\
      --pipeline research \\
      --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \\
      --bucket my-s3-bucket \\
      --region us-east-1

  python sagemaker/launch_training.py \\
      --pipeline dlrm_v3 \\
      --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \\
      --bucket my-s3-bucket \\
      --region us-east-1 \\
      --dataset debug \\
      --mode train
"""

import argparse
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker training job for generative-recommenders."
    )
    parser.add_argument(
        "--pipeline",
        choices=["research", "dlrm_v3"],
        default="research",
        help="Which training pipeline to run.",
    )
    parser.add_argument(
        "--role",
        required=True,
        help="IAM role ARN with SageMaker permissions.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket for output artifacts.",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="AWS region.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="AWS account ID. Resolved automatically via boto3 if not provided.",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.p3.2xlarge",
        dest="instance_type",
        help="SageMaker instance type.",
    )
    parser.add_argument(
        "--image-tag",
        default="latest",
        dest="image_tag",
        help="ECR image tag.",
    )
    # Research-pipeline-specific flags
    parser.add_argument(
        "--gin-config-file",
        default="configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin",
        dest="gin_config_file",
        help="(research) Gin config file path relative to /opt/ml/code/.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ml-1m",
        dest="dataset_name",
        help="(research) Dataset name understood by get_common_preprocessors().",
    )
    parser.add_argument(
        "--master-port",
        default="12345",
        dest="master_port",
        help="(research) DDP master port.",
    )
    # DLRM v3-specific flags
    parser.add_argument(
        "--dataset",
        default="debug",
        help="(dlrm_v3) Dataset key for train_ranker.SUPPORTED_CONFIGS.",
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "train-eval", "streaming-train-eval"],
        help="(dlrm_v3) Training mode.",
    )
    return parser.parse_args()


def get_account_id(account: str | None) -> str:
    if account:
        return account
    import boto3

    return boto3.client("sts").get_caller_identity()["Account"]


def main() -> None:
    args = parse_args()
    account = get_account_id(args.account)

    from sagemaker.estimator import Estimator

    image_uri = (
        f"{account}.dkr.ecr.{args.region}.amazonaws.com/"
        f"generative-recommenders:{args.image_tag}"
    )
    output_path = f"s3://{args.bucket}/generative-recommenders/output/"

    if args.pipeline == "research":
        entry_point_script = "sagemaker/train_research.py"
        hyperparameters = {
            "gin_config_file": args.gin_config_file,
            "dataset_name": args.dataset_name,
            "master_port": args.master_port,
        }
    else:
        entry_point_script = "sagemaker/train_dlrm_v3.py"
        hyperparameters = {
            "dataset": args.dataset,
            "mode": args.mode,
        }

    logger.info(f"Pipeline      : {args.pipeline}")
    logger.info(f"Image URI     : {image_uri}")
    logger.info(f"Instance type : {args.instance_type}")
    logger.info(f"Output path   : {output_path}")
    logger.info(f"Hyperparams   : {hyperparameters}")

    estimator = Estimator(
        image_uri=image_uri,
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        hyperparameters=hyperparameters,
        environment={"SAGEMAKER_PROGRAM": entry_point_script},
        output_path=output_path,
        region_name=args.region,
    )

    estimator.fit()
    logger.info("Training job submitted successfully.")


if __name__ == "__main__":
    main()
