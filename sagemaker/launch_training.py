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

Providing pre-uploaded S3 data (recommended):
  # First prepare the data once:
  python sagemaker/prepare_data.py sample --bucket my-bucket --dataset-name ml-1m
  python sagemaker/prepare_data.py custom --bucket my-bucket \\
      --local-path /path/to/my_data.csv --dataset-name my-dataset

  # Then pass the printed S3 URI to launch_training.py:
  python sagemaker/launch_training.py --pipeline research \\
      --role ... --bucket my-bucket \\
      --data-s3-uri s3://my-bucket/generative-recommenders/data/ml-1m/

MLflow tracking (optional):
  python sagemaker/launch_training.py --pipeline research \\
      --role ... --bucket my-bucket \\
      --data-s3-uri s3://my-bucket/generative-recommenders/data/ml-1m/ \\
      --mlflow-tracking-uri https://<tracking-server-id>.sagemaker.us-east-1.amazonaws.com \\
      --mlflow-experiment-name my-experiment
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
        default="ml.g4dn.xlarge",
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
    # Data channel
    parser.add_argument(
        "--data-s3-uri",
        default=None,
        dest="data_s3_uri",
        help=(
            "S3 URI of the training data folder produced by prepare_data.py "
            "(e.g. s3://my-bucket/generative-recommenders/data/ml-1m/). "
            "If omitted, the training script downloads data at runtime."
        ),
    )
    # MLflow tracking
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        dest="mlflow_tracking_uri",
        help=(
            "MLflow tracking server URI (e.g. the ARN or HTTPS URL of your "
            "SageMaker MLflow Tracking Server). If omitted, MLflow logging is disabled."
        ),
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="generative-recommenders",
        dest="mlflow_experiment_name",
        help="MLflow experiment name.",
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
        entry_point_script = "train_research.py"
        hyperparameters = {
            "gin_config_file": args.gin_config_file,
            "dataset_name": args.dataset_name,
            "master_port": args.master_port,
        }
    else:
        entry_point_script = "train_dlrm_v3.py"
        hyperparameters = {
            "dataset": args.dataset,
            "mode": args.mode,
        }

    logger.info(f"Pipeline      : {args.pipeline}")
    logger.info(f"Image URI     : {image_uri}")
    logger.info(f"Instance type : {args.instance_type}")
    logger.info(f"Output path   : {output_path}")
    logger.info(f"Hyperparams   : {hyperparameters}")
    logger.info(f"Data S3 URI   : {args.data_s3_uri or '(none — will download at runtime)'}")
    logger.info(f"MLflow URI    : {args.mlflow_tracking_uri or '(disabled)'}")

    environment = {"SAGEMAKER_PROGRAM": entry_point_script}
    if args.mlflow_tracking_uri:
        environment["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        environment["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name

    estimator = Estimator(
        image_uri=image_uri,
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        hyperparameters=hyperparameters,
        environment=environment,
        output_path=output_path,
        region_name=args.region,
    )

    inputs = {"training": args.data_s3_uri} if args.data_s3_uri else None
    estimator.fit(inputs)
    logger.info("Training job submitted successfully.")


if __name__ == "__main__":
    main()
