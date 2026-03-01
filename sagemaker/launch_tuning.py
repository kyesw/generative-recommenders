"""
Launch a SageMaker Hyperparameter Tuning job for generative-recommenders.

Usage:
  python sagemaker/launch_tuning.py \
      --pipeline research \
      --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
      --bucket my-s3-bucket \
      --region us-east-1 \
      --data-s3-uri s3://my-bucket/generative-recommenders/data/ml-1m/

Configuration:
  Edit the TUNING_CONFIG section below to define:
  - Which hyperparameters to tune and their ranges
  - The objective metric to optimize
  - Number of training jobs to run
"""

import argparse
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# TUNING CONFIGURATION - EDIT THIS SECTION
# ============================================================================

TUNING_CONFIG = {
    # Objective metric that SageMaker will optimize
    # This must match a metric name that appears in your training logs
    "objective_metric_name": "ndcg@10",
    "objective_type": "Maximize",  # Maximize NDCG@10
    
    # Hyperparameter ranges to search
    # Supported types: ContinuousParameter, IntegerParameter, CategoricalParameter
    "hyperparameter_ranges": {
        # Uncomment the hyperparameters you want to tune:
        "learning_rate": ("Continuous", 0.0001, 0.01),
        "local_batch_size": ("Integer", 64, 256),
        "dropout_rate": ("Continuous", 0.1, 0.5),
        "num_negatives": ("Integer", 64, 256),
        "temperature": ("Continuous", 0.01, 0.1),
    },
    
    # Tuning job limits
    "max_jobs": 20,  # Total number of training jobs to run
    "max_parallel_jobs": 2,  # Number of jobs to run in parallel
    
    # Tuning strategy
    "strategy": "Bayesian",  # Options: "Bayesian", "Random", "Grid", "Hyperband"
    
    # Early stopping (optional)
    "early_stopping_type": "Auto",  # Options: "Off", "Auto"
}

# Metric definitions - SageMaker extracts these from training logs
# Add any metrics you want to track (not just the objective metric)
METRIC_DEFINITIONS = [
    {"Name": "ndcg@10", "Regex": r"NDCG@10 ([0-9\.]+)"},
    {"Name": "ndcg@50", "Regex": r"NDCG@50 ([0-9\.]+)"},
    {"Name": "hr@10", "Regex": r"HR@10 ([0-9\.]+)"},
    {"Name": "hr@50", "Regex": r"HR@50 ([0-9\.]+)"},
    {"Name": "mrr", "Regex": r"MRR ([0-9\.]+)"},
    {"Name": "train_loss", "Regex": r"batch-stat \(train\):.*: ([0-9\.]+)"},
]

# ============================================================================
# END TUNING CONFIGURATION
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker hyperparameter tuning job."
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
    parser.add_argument(
        "--gin-config-file",
        default="configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin",
        dest="gin_config_file",
        help="(research) Gin config file path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ml-1m",
        dest="dataset_name",
        help="(research) Dataset name.",
    )
    parser.add_argument(
        "--master-port",
        default="12345",
        dest="master_port",
        help="(research) DDP master port.",
    )
    parser.add_argument(
        "--data-s3-uri",
        default=None,
        dest="data_s3_uri",
        help="S3 URI of training data (recommended for tuning).",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        dest="mlflow_tracking_uri",
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="generative-recommenders-tuning",
        dest="mlflow_experiment_name",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--dataset",
        default="debug",
        help="(dlrm_v3) Dataset key.",
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "train-eval", "streaming-train-eval"],
        help="(dlrm_v3) Training mode.",
    )
    parser.add_argument(
        "--tuning-job-name",
        default=None,
        dest="tuning_job_name",
        help="Custom name for the tuning job (auto-generated if not provided).",
    )
    return parser.parse_args()


def get_account_id(account: str | None) -> str:
    if account:
        return account
    import boto3
    return boto3.client("sts").get_caller_identity()["Account"]


def build_hyperparameter_ranges():
    """Convert TUNING_CONFIG ranges to SageMaker parameter objects."""
    from sagemaker.tuner import (
        ContinuousParameter,
        IntegerParameter,
        CategoricalParameter,
    )
    
    ranges = {}
    for param_name, config in TUNING_CONFIG["hyperparameter_ranges"].items():
        param_type = config[0]
        if param_type == "Continuous":
            ranges[param_name] = ContinuousParameter(config[1], config[2])
        elif param_type == "Integer":
            ranges[param_name] = IntegerParameter(config[1], config[2])
        elif param_type == "Categorical":
            ranges[param_name] = CategoricalParameter(config[1])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    return ranges


def main() -> None:
    args = parse_args()
    account = get_account_id(args.account)

    from sagemaker.estimator import Estimator
    from sagemaker.tuner import HyperparameterTuner

    # Validate tuning configuration
    if not TUNING_CONFIG["hyperparameter_ranges"]:
        logger.error(
            "No hyperparameter ranges defined in TUNING_CONFIG. "
            "Please edit the TUNING_CONFIG section in this script."
        )
        sys.exit(1)

    image_uri = (
        f"{account}.dkr.ecr.{args.region}.amazonaws.com/"
        f"generative-recommenders:{args.image_tag}"
    )
    output_path = f"s3://{args.bucket}/generative-recommenders/tuning-output/"

    # Build base hyperparameters (fixed values)
    if args.pipeline == "research":
        entry_point_script = "train_research.py"
        base_hyperparameters = {
            "gin_config_file": args.gin_config_file,
            "dataset_name": args.dataset_name,
            "master_port": args.master_port,
        }
    else:
        entry_point_script = "train_dlrm_v3.py"
        base_hyperparameters = {
            "dataset": args.dataset,
            "mode": args.mode,
        }

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING JOB CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Pipeline           : {args.pipeline}")
    logger.info(f"Image URI          : {image_uri}")
    logger.info(f"Instance type      : {args.instance_type}")
    logger.info(f"Output path        : {output_path}")
    logger.info(f"Data S3 URI        : {args.data_s3_uri or '(none)'}")
    logger.info(f"Base hyperparams   : {base_hyperparameters}")
    logger.info(f"Objective metric   : {TUNING_CONFIG['objective_metric_name']} ({TUNING_CONFIG['objective_type']})")
    logger.info(f"Tuning strategy    : {TUNING_CONFIG['strategy']}")
    logger.info(f"Max jobs           : {TUNING_CONFIG['max_jobs']}")
    logger.info(f"Max parallel jobs  : {TUNING_CONFIG['max_parallel_jobs']}")
    logger.info(f"Hyperparameter ranges:")
    for param, config in TUNING_CONFIG["hyperparameter_ranges"].items():
        logger.info(f"  - {param}: {config}")
    logger.info("=" * 80)

    environment = {}
    if args.mlflow_tracking_uri:
        environment["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        environment["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name

    # Create the base estimator
    estimator = Estimator(
        image_uri=image_uri,
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        hyperparameters=base_hyperparameters,
        environment=environment,
        output_path=output_path,
        metric_definitions=METRIC_DEFINITIONS,
        region_name=args.region,
        source_dir=_REPO_ROOT,
        entry_point=entry_point_script,
    )

    # Build hyperparameter ranges
    hyperparameter_ranges = build_hyperparameter_ranges()

    # Create the tuner
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=TUNING_CONFIG["objective_metric_name"],
        objective_type=TUNING_CONFIG["objective_type"],
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=METRIC_DEFINITIONS,
        max_jobs=TUNING_CONFIG["max_jobs"],
        max_parallel_jobs=TUNING_CONFIG["max_parallel_jobs"],
        strategy=TUNING_CONFIG["strategy"],
        early_stopping_type=TUNING_CONFIG["early_stopping_type"],
        base_tuning_job_name=args.tuning_job_name,
    )

    # Launch the tuning job
    inputs = {"training": args.data_s3_uri} if args.data_s3_uri else None
    
    logger.info("Launching hyperparameter tuning job...")
    tuner.fit(inputs, wait=False)
    
    logger.info("=" * 80)
    logger.info("Tuning job submitted successfully!")
    logger.info(f"Job name: {tuner.latest_tuning_job.name}")
    logger.info("Monitor progress in the SageMaker console or use:")
    logger.info(f"  tuner.describe()")
    logger.info(f"  tuner.analytics().dataframe()")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
