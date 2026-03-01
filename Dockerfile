# Dockerfile
# SageMaker-compatible image for both training jobs and real-time inference endpoints.
#
# Training:   SageMaker runs `docker run <image> train`
#             → sagemaker-training-toolkit reads SAGEMAKER_PROGRAM and executes it.
#
# Inference:  SageMaker runs `docker run <image> serve`
#             → gunicorn serves docker/sagemaker_handler.py (Flask) on port 8080,
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
#   ca-certificates  — TLS root certs for HTTPS calls to AWS service endpoints
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        libgomp1 \
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
#   sagemaker-training — provides the `train` CLI entry point; reads the
#                        SAGEMAKER_PROGRAM env var and executes the script
#                        under /opt/ml/code/ with SM_* env vars injected.
#
#   sagemaker (SDK)    — Python SDK required for the Feature Store high-level
#                        API (sagemaker.feature_store.feature_group.FeatureGroup)
#                        used by feature_store_setup.py and ingest_features.py.
#                        Also resolves SageMaker MLflow tracking server credentials.
#
#   boto3              — direct AWS API calls: sagemaker-featurestore-runtime
#                        (online store reads in inference.py), S3, STS, etc.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        sagemaker-training \
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
# Install the generative-recommenders package (compiles CUDA extensions).
# WORKDIR /opt/ml/code is the standard SageMaker code directory; source_dir
# uploads from launch_training.py / deploy_endpoint.py land here at runtime.
# ---------------------------------------------------------------------------
WORKDIR /opt/ml/code
COPY setup.py requirements.txt README.md ./
COPY generative_recommenders generative_recommenders/
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Inference handler (Flask + gunicorn).
# Placed outside /opt/ml/code so runtime source_dir uploads never overwrite it.
# sagemaker_handler.py loads SAGEMAKER_PROGRAM at startup, calls model_fn once,
# then serves /ping and /invocations without any TorchServe dependency.
# ---------------------------------------------------------------------------
COPY docker/sagemaker_handler.py /opt/program/sagemaker_handler.py

# ---------------------------------------------------------------------------
# Entrypoint script
#   train  → exec train         delegates to sagemaker-training-toolkit
#   serve  → gunicorn           custom Flask handler on port 8080
#   *      → pass-through       for local development / custom commands
#
# --preload runs model_fn in the gunicorn master before forking workers so
# model weights are loaded once and shared via copy-on-write.
# exec replaces the shell so gunicorn receives SIGTERM directly.
# ---------------------------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

# Default training entry point; overridden at job launch time via entry_point.
ENV SAGEMAKER_PROGRAM=train_research.py

# /opt/ml/code is on PYTHONPATH so inference.py can import generative_recommenders
# whether the module was loaded from the baked-in package or a source_dir upload.
ENV PYTHONPATH=/opt/ml/code

# Forward log level to the SageMaker toolkit loggers (20 = logging.INFO).
ENV SAGEMAKER_CONTAINER_LOG_LEVEL=20

# ---------------------------------------------------------------------------
# Port 8080 — gunicorn binds here; SageMaker routes inference traffic to it.
# ---------------------------------------------------------------------------
EXPOSE 8080

# ---------------------------------------------------------------------------
# Health check (local Docker / CI; SageMaker polls /ping directly via HTTP)
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8080/ping || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["train"]
