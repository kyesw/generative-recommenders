# Dockerfile_new
# SageMaker-compatible image for both training jobs and real-time inference endpoints.
#
# Training:   SageMaker runs `docker run <image> train`
#             → sagemaker-training-toolkit reads SAGEMAKER_PROGRAM and executes it.
#
# Inference:  SageMaker runs `docker run <image> serve`
#             → sagemaker-inference-toolkit starts nginx (port 8080) + gunicorn,
#               routing GET /ping (health check) and POST /invocations (predict).
#
# Integrations:
#   - SageMaker Feature Store  : boto3 sagemaker-featurestore-runtime client used
#                                by inference.py to fetch user interaction sequences
#                                from the online store at request time.
#   - MLflow on SageMaker      : mlflow + sagemaker-mlflow plugin; the plugin maps
#                                a SageMaker MLflow tracking server ARN/URL to an
#                                HTTPS endpoint and handles SigV4 auth automatically.
#                                Set MLFLOW_TRACKING_URI at job launch time.

# ---------------------------------------------------------------------------
# Base: PyTorch 2.6.0 + CUDA 12.4 + cuDNN 9 (matches requirements.txt)
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# ---------------------------------------------------------------------------
# System packages
#   build-essential  — C/C++ toolchain for compiling CUDA extensions
#   git              — needed by pip editable installs and setup.py
#   wget             — optional dataset download fallback in train_research.py
#   curl             — used by the HEALTHCHECK below and general debugging
#   libgomp1         — OpenMP runtime required by fbgemm-gpu / torchrec
#   nginx            — reverse proxy managed by sagemaker-inference-toolkit
#   ca-certificates  — TLS root certs for HTTPS calls to AWS service endpoints
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        libgomp1 \
        nginx \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# PyTorch ecosystem — CUDA 12.4 wheels
#   fbgemm-gpu  — optimised embedding and sparse ops
#   torchrec    — recommendation-system primitives built on fbgemm-gpu
# Both must come from the PyTorch CUDA 12.4 wheel index; the default PyPI
# builds are CPU-only and will silently degrade performance.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        "fbgemm-gpu>=1.1.0" \
        "torchrec>=1.1.0" \
        --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# SageMaker toolkits + AWS SDKs
#
#   sagemaker-training  — provides the `train` CLI entry point; reads the
#                         SAGEMAKER_PROGRAM env var and executes the script
#                         under /opt/ml/code/ with SM_* env vars injected.
#
#   sagemaker-inference — provides the `serve` CLI entry point; manages the
#                         nginx + gunicorn stack and routes /ping and
#                         /invocations to model_fn / predict_fn / output_fn.
#
#   sagemaker (SDK)     — Python SDK required for the Feature Store high-level
#                         API (sagemaker.feature_store.feature_group.FeatureGroup)
#                         used by feature_store_setup.py and ingest_features.py.
#                         Also resolves SageMaker MLflow tracking server credentials.
#
#   boto3               — direct AWS API calls: sagemaker-featurestore-runtime
#                         (online store reads in inference.py), S3, STS, etc.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        sagemaker-training \
        sagemaker-inference \
        "sagemaker>=2.200.0" \
        boto3

# ---------------------------------------------------------------------------
# MLflow experiment tracking
#
#   mlflow           — core library; train_research.py and train_dlrm_v3.py
#                      log metrics, params and artifacts via the standard
#                      mlflow.* API calls.
#
#   sagemaker-mlflow — plugin that activates automatically when
#                      MLFLOW_TRACKING_URI points to a SageMaker managed
#                      MLflow tracking server (ARN or HTTPS endpoint).
#                      Injects SigV4-signed requests so no extra auth code is
#                      needed; credentials come from the SageMaker execution role.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        "mlflow>=2.11.0" \
        sagemaker-mlflow

# ---------------------------------------------------------------------------
# Application Python dependencies
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        "gin-config>=0.5.0" \
        "pandas>=2.2.0" \
        "tensorboard>=2.19.0" \
        pybind11 \
        click \
        matplotlib \
        absl-py \
        flask \
        gunicorn \
        gevent

# ---------------------------------------------------------------------------
# Install the generative-recommenders package (compiles CUDA extensions)
# WORKDIR /opt/ml/code is the standard SageMaker code directory; the training
# toolkit appends it to PYTHONPATH automatically and source_dir uploads land here.
# ---------------------------------------------------------------------------
WORKDIR /opt/ml/code
COPY setup.py requirements.txt README.md ./
COPY generative_recommenders generative_recommenders/
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Entrypoint script
#   train  → exec train    delegates to sagemaker-training-toolkit
#   serve  → exec serve    delegates to sagemaker-inference-toolkit
#   *      → pass-through  for local development / custom commands
#
# Using exec ensures the child process (train/serve) receives OS signals
# (SIGTERM / SIGKILL) directly — required by SageMaker's graceful shutdown
# protocol (SIGTERM followed by SIGKILL after 120 seconds during training).
# ---------------------------------------------------------------------------
RUN printf '%s\n' \
        '#!/bin/bash' \
        'set -e' \
        'case "$1" in' \
        '    train) exec train ;;' \
        '    serve) exec serve ;;' \
        '    *)     exec "$@" ;;' \
        'esac' \
    > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

# Default training entry point; overridden at job launch time when running
# train_dlrm_v3.py (the Estimator passes source_dir + entry_point instead).
ENV SAGEMAKER_PROGRAM=train_research.py

# Make /opt/ml/code importable at both training (source_dir upload) and
# inference (model_dir layout). sagemaker-training-toolkit also sets this,
# but explicit declaration ensures it is set for `serve` containers too.
ENV PYTHONPATH=/opt/ml/code

# Forward log level to the SageMaker toolkit loggers (20 = logging.INFO).
# Increase to 10 (DEBUG) for verbose toolkit diagnostics.
ENV SAGEMAKER_CONTAINER_LOG_LEVEL=20

# ---------------------------------------------------------------------------
# Port — sagemaker-inference-toolkit starts nginx on 8080.
# SageMaker only routes traffic to this port for real-time endpoints.
# ---------------------------------------------------------------------------
EXPOSE 8080

# ---------------------------------------------------------------------------
# Health check (used by Docker and local testing; SageMaker polls /ping directly)
#   start-period  — allow up to 120 s for model loading before the first check
#   interval      — re-check every 30 s
#   timeout       — fail a check if no response within 10 s
#   retries       — mark unhealthy after 3 consecutive failures
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8080/ping || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["train"]
