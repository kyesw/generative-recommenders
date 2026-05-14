"""
SageMaker Pipeline for batch inference with pre/post processing.

Creates a 3-step pipeline visible in SageMaker Studio:
  1. Preprocess  — Query database, write JSONL to S3
  2. Transform   — Run Batch Transform (JSONL -> recommendations)
  3. Postprocess — Read results, write back to database

Model selection:
  - Provide --mlflow-run-id to pin a specific model version.
  - Omit it to auto-select the latest successful run from the
    MLflow experiment specified by --mlflow-experiment-name.

Scaling:
  - --transform-instance-count controls how many GPU instances
    run in parallel. The preprocessing step automatically splits
    the input into one file per instance so SageMaker can
    distribute the work evenly.

Usage:
  # Run with a specific model:
  python sagemaker/08_batch_pipeline.py \\
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \\
      --bucket my-bucket \\
      --mlflow-tracking-uri arn:aws:sagemaker:REGION:ACCOUNT:mlflow-app/app-XXX \\
      --mlflow-run-id <run-id>

  # Auto-select latest model, 2 parallel instances:
  python sagemaker/08_batch_pipeline.py \\
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \\
      --bucket my-bucket \\
      --mlflow-tracking-uri arn:aws:sagemaker:REGION:ACCOUNT:mlflow-app/app-XXX \\
      --mlflow-experiment-name my-experiment \\
      --transform-instance-count 2

  # Create pipeline definition only (don't execute):
  python sagemaker/08_batch_pipeline.py \\
      --role ... --bucket ... --mlflow-tracking-uri ... \\
      --mlflow-run-id ... --create-only
"""

import argparse
import logging
import os
import sys
import time

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# CLI arguments
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create and run a SageMaker Pipeline for batch inference."
    )

    # --- Required --------------------------------------------------------
    p.add_argument("--role", required=True, help="IAM role ARN for SageMaker.")
    p.add_argument("--bucket", required=True, help="S3 bucket for pipeline artifacts.")
    p.add_argument(
        "--mlflow-tracking-uri", required=True, dest="mlflow_tracking_uri",
        help="MLflow tracking URI (SageMaker MLflow app ARN).",
    )

    # --- Model selection -------------------------------------------------
    p.add_argument(
        "--mlflow-run-id", default=None, dest="mlflow_run_id",
        help="MLflow run ID. If omitted, the latest successful run is used.",
    )
    p.add_argument(
        "--mlflow-experiment-name", default="generative-recommenders",
        dest="mlflow_experiment_name",
        help="MLflow experiment to search when --mlflow-run-id is omitted.",
    )

    # --- Infrastructure --------------------------------------------------
    p.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"),
    )
    p.add_argument(
        "--account", default=None,
        help="AWS account ID. Resolved via STS if not provided.",
    )
    p.add_argument("--image-tag", default="latest", dest="image_tag")
    p.add_argument(
        "--pipeline-name", default="gr-batch-inference", dest="pipeline_name",
    )
    p.add_argument(
        "--transform-instance-count", type=int, default=1,
        dest="transform_instance_count",
        help=(
            "Number of Batch Transform instances. Input is auto-split into "
            "one file per instance for parallel processing."
        ),
    )
    p.add_argument(
        "--transform-instance-type", default="ml.g4dn.xlarge",
        dest="transform_instance_type",
    )
    p.add_argument(
        "--processing-instance-type", default="ml.m5.large",
        dest="processing_instance_type",
    )

    # --- Inference parameters --------------------------------------------
    p.add_argument("--top-k", type=int, default=20, dest="top_k",
                   help="Number of recommendations per user.")
    p.add_argument(
        "--num-users", type=int, default=100, dest="num_users",
        help="Number of users to process (passed as --limit to preprocess).",
    )
    p.add_argument("--dataset-name", default="ml-1m", dest="dataset_name")
    p.add_argument(
        "--db-connection-string", default="mock://localhost",
        dest="db_connection_string",
        help="Database URI. Use mock://localhost for testing.",
    )

    # --- Execution control -----------------------------------------------
    p.add_argument(
        "--create-only", action="store_true", dest="create_only",
        help="Upsert pipeline definition without starting an execution.",
    )
    p.add_argument("--wait", action="store_true", default=True, dest="wait")
    p.add_argument("--no-wait", action="store_false", dest="wait")

    return p.parse_args()


# =====================================================================
# Helpers
# =====================================================================

def get_account_id(account, region):
    if account:
        return account
    import boto3
    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def resolve_mlflow_run_id(tracking_uri, experiment_name):
    """Return the run ID of the latest successful MLflow run."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{experiment_name}' not found. "
            f"List experiments with: mlflow.search_experiments()"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            f"No successful runs in experiment '{experiment_name}'."
        )

    run_id = runs.iloc[0]["run_id"]
    logger.info(
        f"Resolved latest MLflow run: {run_id} "
        f"(experiment='{experiment_name}', started={runs.iloc[0]['start_time']})"
    )
    return run_id


# =====================================================================
# Pipeline definition
# =====================================================================

def main() -> None:
    args = parse_args()

    import boto3
    import sagemaker
    import sagemaker.sklearn
    from sagemaker.model import Model
    from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
    from sagemaker.transformer import Transformer
    from sagemaker.workflow.execution_variables import ExecutionVariables
    from sagemaker.workflow.functions import Join
    from sagemaker.workflow.parameters import ParameterInteger, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.steps import ProcessingStep, TransformStep

    # --- Sessions --------------------------------------------------------
    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)
    pipeline_session = PipelineSession(boto_session=boto_session)

    # --- Resolve model ---------------------------------------------------
    if not args.mlflow_run_id:
        args.mlflow_run_id = resolve_mlflow_run_id(
            args.mlflow_tracking_uri, args.mlflow_experiment_name
        )

    account = get_account_id(args.account, args.region)
    image_uri = (
        f"{account}.dkr.ecr.{args.region}.amazonaws.com/"
        f"generative-recommenders:{args.image_tag}"
    )
    s3_base = f"s3://{args.bucket}/generative-recommenders/batch-pipeline"
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

    # --- Pipeline parameters (overridable at execution time) -------------
    param_top_k = ParameterInteger(name="TopK", default_value=args.top_k)
    param_num_users = ParameterInteger(name="NumUsers", default_value=args.num_users)
    param_dataset_name = ParameterString(name="DatasetName", default_value=args.dataset_name)
    param_db_connection = ParameterString(
        name="DbConnectionString", default_value=args.db_connection_string,
    )
    param_transform_instance = ParameterString(
        name="TransformInstanceType", default_value=args.transform_instance_type,
    )
    param_processing_instance = ParameterString(
        name="ProcessingInstanceType", default_value=args.processing_instance_type,
    )
    param_num_splits = ParameterInteger(
        name="NumSplits", default_value=args.transform_instance_count,
    )

    exec_id = ExecutionVariables.PIPELINE_EXECUTION_ID

    # -----------------------------------------------------------------
    # Step 1: Preprocess — query DB, write JSONL to S3
    # -----------------------------------------------------------------
    framework_processor = FrameworkProcessor(
        estimator_cls=sagemaker.sklearn.SKLearn,
        framework_version="1.2-1",
        instance_type=args.processing_instance_type,
        instance_count=1,
        role=args.role,
        sagemaker_session=pipeline_session,
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        step_args=framework_processor.run(
            code="preprocess.py",
            source_dir=scripts_dir,
            outputs=[
                ProcessingOutput(
                    output_name="jsonl",
                    source="/opt/ml/processing/output/jsonl",
                    destination=Join(
                        on="/", values=[s3_base, "preprocessing", exec_id, "output"],
                    ),
                ),
            ],
            arguments=[
                "--db-connection-string", param_db_connection,
                "--dataset-name", param_dataset_name,
                "--top-k", param_top_k.to_string(),
                "--limit", param_num_users.to_string(),
                "--num-splits", param_num_splits.to_string(),
            ],
        ),
    )

    # -----------------------------------------------------------------
    # Create Model (outside the pipeline — reused across executions)
    # -----------------------------------------------------------------
    model_name = f"gr-pipeline-model-{int(time.time())}"
    Model(
        image_uri=image_uri,
        model_data=None,
        role=args.role,
        name=model_name,
        env={
            "SAGEMAKER_PROGRAM": "inference.py",
            "MLFLOW_TRACKING_URI": args.mlflow_tracking_uri,
            "MLFLOW_RUN_ID": args.mlflow_run_id,
            "FEATURE_STORE_REGION": args.region,
        },
        sagemaker_session=sm_session,
    ).create(instance_type=args.transform_instance_type)
    logger.info(f"Model created: {model_name}")

    # -----------------------------------------------------------------
    # Step 2: Batch Transform — run inference on JSONL
    # -----------------------------------------------------------------
    transformer = Transformer(
        model_name=model_name,
        instance_count=args.transform_instance_count,
        instance_type=args.transform_instance_type,
        output_path=Join(on="/", values=[s3_base, "transform-output", exec_id]),
        strategy="SingleRecord",
        assemble_with="Line",
        max_concurrent_transforms=1,
        max_payload=1,
        sagemaker_session=pipeline_session,
    )

    preprocess_output_uri = step_preprocess.properties.ProcessingOutputConfig.Outputs[
        "jsonl"
    ].S3Output.S3Uri

    step_transform = TransformStep(
        name="BatchTransform",
        step_args=transformer.transform(
            data=preprocess_output_uri,
            content_type="application/json",
            split_type="Line",
        ),
    )

    # -----------------------------------------------------------------
    # Step 3: Postprocess — read results, write to DB
    # -----------------------------------------------------------------
    step_postprocess = ProcessingStep(
        name="Postprocess",
        step_args=framework_processor.run(
            code="postprocess.py",
            source_dir=scripts_dir,
            inputs=[
                ProcessingInput(
                    input_name="transform_output",
                    source=step_transform.properties.TransformOutput.S3OutputPath,
                    destination="/opt/ml/processing/input/transform_output",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="results",
                    source="/opt/ml/processing/output/results",
                    destination=Join(
                        on="/", values=[s3_base, "postprocessing", exec_id, "output"],
                    ),
                ),
            ],
            arguments=[
                "--db-connection-string", param_db_connection,
            ],
        ),
    )

    # -----------------------------------------------------------------
    # Assemble and upsert pipeline
    # -----------------------------------------------------------------
    pipeline = Pipeline(
        name=args.pipeline_name,
        parameters=[
            param_top_k,
            param_num_users,
            param_dataset_name,
            param_db_connection,
            param_transform_instance,
            param_processing_instance,
            param_num_splits,
        ],
        steps=[step_preprocess, step_transform, step_postprocess],
        sagemaker_session=pipeline_session,
    )

    logger.info(f"Pipeline       : {args.pipeline_name}")
    logger.info(f"Model          : {model_name}")
    logger.info(f"Image URI      : {image_uri}")
    logger.info(f"MLflow run ID  : {args.mlflow_run_id}")
    logger.info(f"S3 base        : {s3_base}")
    logger.info(f"DB connection  : {args.db_connection_string}")
    logger.info(f"Instances      : {args.transform_instance_count}x {args.transform_instance_type}")

    pipeline.upsert(role_arn=args.role)
    logger.info("Pipeline upserted successfully.")

    if args.create_only:
        logger.info("--create-only: pipeline created/updated but not started.")
        return

    # -----------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------
    execution = pipeline.start()
    logger.info(f"Pipeline execution started: {execution.arn}")

    if args.wait:
        logger.info("Waiting for pipeline execution to complete...")
        execution.wait()
        status = execution.describe()["PipelineExecutionStatus"]
        logger.info(f"Pipeline execution finished: {status}")

        if status != "Succeeded":
            for step in execution.list_steps():
                if step.get("StepStatus") == "Failed":
                    logger.error(
                        f"Step '{step['StepName']}' failed: "
                        f"{step.get('FailureReason', 'unknown')}"
                    )
    else:
        logger.info(
            f"Pipeline running in background.\n"
            f"  Monitor: https://{args.region}.console.aws.amazon.com/sagemaker/home"
            f"?region={args.region}#/pipelines/{args.pipeline_name}"
        )


if __name__ == "__main__":
    main()
