#!/usr/bin/env bash
# Triggers an AWS CodeBuild build to build and push the Docker image to ECR.
# Creates the IAM role and CodeBuild project automatically if they do not exist.
# The actual build steps are defined in buildspec.yml at the repo root.
#
# Required env vars:
#   GITHUB_REPO_URL      HTTPS URL of your GitHub fork
#                        e.g. https://github.com/YOUR_USERNAME/generative-recommenders
#
# Optional env vars:
#   AWS_ACCOUNT_ID       defaults to current caller identity
#   AWS_REGION           defaults to ap-northeast-2
#   CODEBUILD_ROLE_NAME  defaults to generative-recommenders-codebuild-role
#   CODEBUILD_PROJECT    defaults to generative-recommenders-build
#   IMAGE_NAME           defaults to generative-recommenders
#   IMAGE_TAG            defaults to latest
#
# Usage:
#   GITHUB_REPO_URL=https://github.com/YOUR_USERNAME/generative-recommenders \
#     bash sagemaker/build_and_push.sh [--no-wait]

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
CODEBUILD_ROLE_NAME="${CODEBUILD_ROLE_NAME:-generative-recommenders-codebuild-role}"
CODEBUILD_PROJECT="${CODEBUILD_PROJECT:-generative-recommenders-build}"
IMAGE_NAME="${IMAGE_NAME:-generative-recommenders}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NO_WAIT="${1:-}"

: "${GITHUB_REPO_URL:?GITHUB_REPO_URL env var is required}"

# ---------------------------------------------------------------------------
# Create IAM role for CodeBuild if it doesn't exist
# ---------------------------------------------------------------------------
if aws iam get-role --role-name "${CODEBUILD_ROLE_NAME}" > /dev/null 2>&1; then
    echo "==> IAM role '${CODEBUILD_ROLE_NAME}' already exists."
    CODEBUILD_ROLE_ARN=$(aws iam get-role \
        --role-name "${CODEBUILD_ROLE_NAME}" \
        --query "Role.Arn" --output text)
else
    echo "==> Creating IAM role '${CODEBUILD_ROLE_NAME}'..."

    TRUST_POLICY='{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": { "Service": "codebuild.amazonaws.com" },
            "Action": "sts:AssumeRole"
        }]
    }'

    CODEBUILD_ROLE_ARN=$(aws iam create-role \
        --role-name "${CODEBUILD_ROLE_NAME}" \
        --assume-role-policy-document "${TRUST_POLICY}" \
        --query "Role.Arn" --output text)

    PERMISSIONS_POLICY='{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:PutImage",
                    "ecr:DescribeRepositories",
                    "ecr:CreateRepository"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "*"
            }
        ]
    }'

    aws iam put-role-policy \
        --role-name "${CODEBUILD_ROLE_NAME}" \
        --policy-name "${CODEBUILD_ROLE_NAME}-policy" \
        --policy-document "${PERMISSIONS_POLICY}"

    echo "==> IAM role created: ${CODEBUILD_ROLE_ARN}"
    echo "==> Waiting for IAM role to propagate..."
    sleep 10
fi

# ---------------------------------------------------------------------------
# Create CodeBuild project if it doesn't exist
# ---------------------------------------------------------------------------
if aws codebuild batch-get-projects --names "${CODEBUILD_PROJECT}" --region "${AWS_REGION}" \
        --query "projects[0].name" --output text 2>/dev/null | grep -q "${CODEBUILD_PROJECT}"; then
    echo "==> CodeBuild project '${CODEBUILD_PROJECT}' already exists."
else
    echo "==> Creating CodeBuild project '${CODEBUILD_PROJECT}'..."
    aws codebuild create-project \
        --region "${AWS_REGION}" \
        --name "${CODEBUILD_PROJECT}" \
        --source "{
            \"type\": \"GITHUB\",
            \"location\": \"${GITHUB_REPO_URL}\",
            \"buildspec\": \"buildspec.yml\",
            \"gitCloneDepth\": 1
        }" \
        --artifacts "{\"type\": \"NO_ARTIFACTS\"}" \
        --environment "{
            \"type\": \"LINUX_CONTAINER\",
            \"image\": \"aws/codebuild/standard:7.0\",
            \"computeType\": \"BUILD_GENERAL1_LARGE\",
            \"privilegedMode\": true,
            \"environmentVariables\": [
                {\"name\": \"AWS_ACCOUNT_ID\", \"value\": \"${AWS_ACCOUNT_ID}\", \"type\": \"PLAINTEXT\"},
                {\"name\": \"AWS_REGION\",     \"value\": \"${AWS_REGION}\",     \"type\": \"PLAINTEXT\"},
                {\"name\": \"IMAGE_NAME\",     \"value\": \"${IMAGE_NAME}\",     \"type\": \"PLAINTEXT\"},
                {\"name\": \"IMAGE_TAG\",      \"value\": \"${IMAGE_TAG}\",      \"type\": \"PLAINTEXT\"}
            ]
        }" \
        --service-role "${CODEBUILD_ROLE_ARN}" \
        --logs-config "{
            \"cloudWatchLogs\": {
                \"status\": \"ENABLED\",
                \"groupName\": \"/aws/codebuild/${CODEBUILD_PROJECT}\"
            }
        }"
    echo "==> Project created."
fi

# ---------------------------------------------------------------------------
# Start build
# ---------------------------------------------------------------------------
echo "==> Starting build on project: ${CODEBUILD_PROJECT}"
BUILD_ID=$(aws codebuild start-build \
    --project-name "${CODEBUILD_PROJECT}" \
    --region "${AWS_REGION}" \
    --query "build.id" \
    --output text)

echo "==> Build started: ${BUILD_ID}"
echo "    Console: https://${AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${CODEBUILD_PROJECT}/build/${BUILD_ID}/log"

if [ "${NO_WAIT}" = "--no-wait" ]; then
    echo "==> --no-wait passed; exiting. Check the console for build status."
    exit 0
fi

# ---------------------------------------------------------------------------
# Poll until done
# ---------------------------------------------------------------------------
echo "==> Polling for build completion (every 30s)..."
while true; do
    STATUS=$(aws codebuild batch-get-builds \
        --ids "${BUILD_ID}" \
        --region "${AWS_REGION}" \
        --query "builds[0].buildStatus" \
        --output text)

    echo "    Status: ${STATUS}"

    case "${STATUS}" in
        SUCCEEDED)
            echo "==> Build succeeded."
            exit 0
            ;;
        FAILED|FAULT|TIMED_OUT|STOPPED)
            echo "==> Build failed with status: ${STATUS}"
            exit 1
            ;;
        *)
            sleep 30
            ;;
    esac
done
