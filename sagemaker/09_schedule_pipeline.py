"""
Schedule the batch inference pipeline using Amazon EventBridge.

Creates an EventBridge rule that triggers the SageMaker Pipeline on a
cron or rate schedule. Also creates the IAM role that allows EventBridge
to start pipeline executions.

Usage:
  # Run daily at 03:00 UTC:
  python sagemaker/09_schedule_pipeline.py enable \
      --pipeline-name gr-batch-inference \
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
      --schedule "cron(0 3 * * ? *)" \
      --region ap-northeast-2

  # Run every 6 hours:
  python sagemaker/09_schedule_pipeline.py enable \
      --pipeline-name gr-batch-inference \
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
      --schedule "rate(6 hours)" \
      --region ap-northeast-2

  # Override pipeline parameters at execution time:
  python sagemaker/09_schedule_pipeline.py enable \
      --pipeline-name gr-batch-inference \
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
      --schedule "rate(1 day)" \
      --pipeline-param TopK=50 \
      --pipeline-param NumUsers=500

  # Disable the schedule:
  python sagemaker/09_schedule_pipeline.py disable \
      --pipeline-name gr-batch-inference \
      --region ap-northeast-2

  # Check current schedule status:
  python sagemaker/09_schedule_pipeline.py status \
      --pipeline-name gr-batch-inference \
      --region ap-northeast-2
"""

import argparse
import json
import logging
import sys

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

_RULE_NAME_TPL = "{pipeline_name}-schedule"
_EVENTBRIDGE_ROLE_NAME_TPL = "{pipeline_name}-eventbridge-role"


def _rule_name(pipeline_name: str) -> str:
    return _RULE_NAME_TPL.format(pipeline_name=pipeline_name)


def _eventbridge_role_name(pipeline_name: str) -> str:
    return _EVENTBRIDGE_ROLE_NAME_TPL.format(pipeline_name=pipeline_name)


def _get_account_id(region: str) -> str:
    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def _ensure_eventbridge_role(pipeline_name: str, pipeline_arn: str, region: str) -> str:
    """Create or retrieve the IAM role that allows EventBridge to start the pipeline."""
    iam = boto3.client("iam")
    role_name = _eventbridge_role_name(pipeline_name)

    try:
        role = iam.get_role(RoleName=role_name)
        role_arn = role["Role"]["Arn"]
        logger.info(f"IAM role '{role_name}' already exists.")
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        pass

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "events.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=f"Allows EventBridge to start SageMaker Pipeline '{pipeline_name}'.",
    )
    role_arn = role["Role"]["Arn"]

    permissions_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sagemaker:StartPipelineExecution",
                "Resource": pipeline_arn,
            }
        ],
    }

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=f"{role_name}-policy",
        PolicyDocument=json.dumps(permissions_policy),
    )

    logger.info(f"Created IAM role: {role_arn}")
    return role_arn


def _get_pipeline_arn(pipeline_name: str, region: str) -> str:
    sm = boto3.client("sagemaker", region_name=region)
    resp = sm.describe_pipeline(PipelineName=pipeline_name)
    return resp["PipelineArn"]


def enable(args: argparse.Namespace) -> None:
    events = boto3.client("events", region_name=args.region)
    rule_name = _rule_name(args.pipeline_name)

    pipeline_arn = _get_pipeline_arn(args.pipeline_name, args.region)
    eventbridge_role_arn = _ensure_eventbridge_role(
        args.pipeline_name, pipeline_arn, args.region
    )

    logger.info(f"Creating/updating EventBridge rule: {rule_name}")
    logger.info(f"  Schedule: {args.schedule}")

    events.put_rule(
        Name=rule_name,
        ScheduleExpression=args.schedule,
        State="ENABLED",
        Description=f"Triggers SageMaker Pipeline '{args.pipeline_name}' on schedule.",
    )

    pipeline_params = []
    for param in args.pipeline_params:
        key, value = param.split("=", 1)
        pipeline_params.append({"Name": key, "Value": value})

    target_input = {"PipelineParameterList": pipeline_params} if pipeline_params else {}

    events.put_targets(
        Rule=rule_name,
        Targets=[
            {
                "Id": f"{args.pipeline_name}-target",
                "Arn": pipeline_arn,
                "RoleArn": eventbridge_role_arn,
                "SageMakerPipelineParameters": {
                    "PipelineParameterList": pipeline_params,
                },
            }
        ],
    )

    logger.info(f"EventBridge schedule enabled for '{args.pipeline_name}'.")
    if pipeline_params:
        logger.info(f"  Pipeline parameters: {pipeline_params}")


def disable(args: argparse.Namespace) -> None:
    events = boto3.client("events", region_name=args.region)
    rule_name = _rule_name(args.pipeline_name)

    try:
        events.disable_rule(Name=rule_name)
        logger.info(f"EventBridge rule '{rule_name}' disabled.")
    except events.exceptions.ResourceNotFoundException:
        logger.warning(f"Rule '{rule_name}' not found — nothing to disable.")


def status(args: argparse.Namespace) -> None:
    events = boto3.client("events", region_name=args.region)
    rule_name = _rule_name(args.pipeline_name)

    try:
        rule = events.describe_rule(Name=rule_name)
    except events.exceptions.ResourceNotFoundException:
        logger.info(f"No schedule found for pipeline '{args.pipeline_name}'.")
        return

    logger.info(f"Rule name : {rule['Name']}")
    logger.info(f"State     : {rule['State']}")
    logger.info(f"Schedule  : {rule['ScheduleExpression']}")
    logger.info(f"ARN       : {rule['Arn']}")

    targets = events.list_targets_by_rule(Rule=rule_name)
    for t in targets.get("Targets", []):
        logger.info(f"Target    : {t['Arn']}")
        params = t.get("SageMakerPipelineParameters", {}).get("PipelineParameterList", [])
        if params:
            logger.info(f"  Params  : {params}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schedule a SageMaker Pipeline with EventBridge."
    )
    parser.add_argument("--region", default="ap-northeast-2")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- enable ---
    enable_parser = subparsers.add_parser("enable", help="Enable a schedule.")
    enable_parser.add_argument(
        "--pipeline-name", required=True, dest="pipeline_name",
        help="SageMaker Pipeline name (as created by 08_batch_pipeline.py).",
    )
    enable_parser.add_argument(
        "--role", required=True,
        help="IAM role ARN for SageMaker Pipeline execution.",
    )
    enable_parser.add_argument(
        "--schedule", required=True,
        help="EventBridge schedule expression (e.g. 'rate(1 day)' or 'cron(0 3 * * ? *)').",
    )
    enable_parser.add_argument(
        "--pipeline-param", action="append", default=[], dest="pipeline_params",
        help="Pipeline parameter override as Key=Value. Can be repeated.",
    )

    # --- disable ---
    disable_parser = subparsers.add_parser("disable", help="Disable the schedule.")
    disable_parser.add_argument(
        "--pipeline-name", required=True, dest="pipeline_name",
    )

    # --- status ---
    status_parser = subparsers.add_parser("status", help="Show schedule status.")
    status_parser.add_argument(
        "--pipeline-name", required=True, dest="pipeline_name",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "enable":
        enable(args)
    elif args.command == "disable":
        disable(args)
    elif args.command == "status":
        status(args)


if __name__ == "__main__":
    main()
