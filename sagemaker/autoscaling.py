"""
Enable or disable auto scaling for a SageMaker real-time endpoint.

Usage:
  # Enable auto scaling
  python sagemaker/autoscaling.py enable \
      --endpoint-name generative-recommenders-1234567890 \
      --region ap-northeast-2

  # Disable auto scaling
  python sagemaker/autoscaling.py disable \
      --endpoint-name generative-recommenders-1234567890 \
      --region ap-northeast-2

Optional flags for enable:
  --min-instances   Minimum number of instances (default: 1)
  --max-instances   Maximum number of instances (default: 4)
  --target-rps      Target invocations per instance per minute (default: 50)
  --scale-out-cooldown  Seconds to wait before scaling out again (default: 60)
  --scale-in-cooldown   Seconds to wait before scaling in again (default: 600)
"""

import argparse
import logging
import sys

import boto3

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

_RESOURCE_ID_TPL = "endpoint/{endpoint_name}/variant/AllTraffic"
_SCALABLE_DIMENSION = "sagemaker:variant:DesiredInstanceCount"
_SERVICE_NAMESPACE = "sagemaker"


def _resource_id(endpoint_name: str) -> str:
    return _RESOURCE_ID_TPL.format(endpoint_name=endpoint_name)


def enable(args: argparse.Namespace) -> None:
    aas = boto3.client("application-autoscaling", region_name=args.region)
    resource_id = _resource_id(args.endpoint_name)
    policy_name = f"{args.endpoint_name}-scaling-policy"

    logger.info(f"Registering scalable target: {resource_id}")
    aas.register_scalable_target(
        ServiceNamespace=_SERVICE_NAMESPACE,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIMENSION,
        MinCapacity=args.min_instances,
        MaxCapacity=args.max_instances,
    )

    logger.info(
        f"Creating scaling policy: target={args.target_rps} invocations/instance/min, "
        f"scale-out cooldown={args.scale_out_cooldown}s, "
        f"scale-in cooldown={args.scale_in_cooldown}s"
    )
    aas.put_scaling_policy(
        PolicyName=policy_name,
        ServiceNamespace=_SERVICE_NAMESPACE,
        ResourceId=resource_id,
        ScalableDimension=_SCALABLE_DIMENSION,
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": float(args.target_rps),
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleOutCooldown": args.scale_out_cooldown,
            "ScaleInCooldown": args.scale_in_cooldown,
        },
    )

    logger.info(
        f"Auto scaling enabled on '{args.endpoint_name}' "
        f"[{args.min_instances}–{args.max_instances} instances]"
    )


def disable(args: argparse.Namespace) -> None:
    aas = boto3.client("application-autoscaling", region_name=args.region)
    resource_id = _resource_id(args.endpoint_name)
    policy_name = f"{args.endpoint_name}-scaling-policy"

    # Delete scaling policy first
    try:
        aas.delete_scaling_policy(
            PolicyName=policy_name,
            ServiceNamespace=_SERVICE_NAMESPACE,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIMENSION,
        )
        logger.info(f"Deleted scaling policy: {policy_name}")
    except aas.exceptions.ObjectNotFoundException:
        logger.warning("Scaling policy not found — may already be removed.")

    # Deregister scalable target
    try:
        aas.deregister_scalable_target(
            ServiceNamespace=_SERVICE_NAMESPACE,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIMENSION,
        )
        logger.info(f"Deregistered scalable target: {resource_id}")
    except aas.exceptions.ObjectNotFoundException:
        logger.warning("Scalable target not found — may already be removed.")

    logger.info(f"Auto scaling disabled on '{args.endpoint_name}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enable or disable auto scaling for a SageMaker endpoint."
    )
    parser.add_argument("--region", default="us-east-1")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- enable ---
    enable_parser = subparsers.add_parser("enable", help="Enable auto scaling.")
    enable_parser.add_argument("--endpoint-name", required=True, dest="endpoint_name")
    enable_parser.add_argument("--min-instances", type=int, default=1, dest="min_instances")
    enable_parser.add_argument("--max-instances", type=int, default=4, dest="max_instances")
    enable_parser.add_argument(
        "--target-rps",
        type=int,
        default=50,
        dest="target_rps",
        help="Target invocations per instance per minute.",
    )
    enable_parser.add_argument(
        "--scale-out-cooldown",
        type=int,
        default=60,
        dest="scale_out_cooldown",
        help="Seconds before scaling out again (default: 60).",
    )
    enable_parser.add_argument(
        "--scale-in-cooldown",
        type=int,
        default=600,
        dest="scale_in_cooldown",
        help="Seconds before scaling in again (default: 600).",
    )

    # --- disable ---
    disable_parser = subparsers.add_parser("disable", help="Disable auto scaling.")
    disable_parser.add_argument("--endpoint-name", required=True, dest="endpoint_name")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "enable":
        enable(args)
    elif args.command == "disable":
        disable(args)


if __name__ == "__main__":
    main()
