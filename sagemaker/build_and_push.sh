#!/usr/bin/env bash
set -euo pipefail

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
AWS_REGION="${AWS_REGION:-us-east-1}"
IMAGE_NAME="${IMAGE_NAME:-generative-recommenders}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"
FULL_IMAGE="${ECR_REPO}:${IMAGE_TAG}"

echo "==> Logging in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Creating ECR repository (if it doesn't exist)"
aws ecr describe-repositories --repository-names "${IMAGE_NAME}" --region "${AWS_REGION}" > /dev/null 2>&1 \
    || aws ecr create-repository --repository-name "${IMAGE_NAME}" --region "${AWS_REGION}"

echo "==> Building Docker image"
# Run from repo root so COPY . /opt/ml/code/ picks up everything
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
docker build -t "${IMAGE_NAME}" "${REPO_ROOT}"

echo "==> Tagging image"
docker tag "${IMAGE_NAME}:latest" "${FULL_IMAGE}"

echo "==> Pushing image to ECR"
docker push "${FULL_IMAGE}"

echo ""
echo "Image pushed: ${FULL_IMAGE}"
