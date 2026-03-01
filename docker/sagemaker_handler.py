"""
Minimal SageMaker real-time inference server.

Loads the handler module defined by SAGEMAKER_PROGRAM and serves
  GET  /ping         → 200 OK  (SageMaker health check)
  POST /invocations  → prediction result

The handler module must expose:
    model_fn(model_dir)                    → model context
    input_fn(request_body, content_type)   → parsed input
    predict_fn(data, model)                → prediction
    output_fn(prediction, accept)          → serialised response string

Launched by gunicorn with --preload so model_fn runs once in the
master process before workers fork — model weights are shared via
copy-on-write and not reloaded per worker.
"""

import importlib.util
import logging
import os
import sys
import tarfile

import boto3
from flask import Flask, Response, request

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download and extract source code from S3 if SAGEMAKER_SUBMIT_DIRECTORY is
# set. SageMaker sets this env var for custom inference containers but does
# NOT automatically download the code — that is the toolkit's responsibility.
# We replicate that step here so inference.py and configs/ are available at
# /opt/ml/code/ before model_fn is called.
# ---------------------------------------------------------------------------
_submit_dir = os.environ.get("SAGEMAKER_SUBMIT_DIRECTORY", "")
if _submit_dir.startswith("s3://"):
    _s3_path = _submit_dir[len("s3://"):]
    _bucket, _key = _s3_path.split("/", 1)
    _tmp = "/tmp/sourcedir.tar.gz"
    _code_dir = "/opt/ml/code"
    logger.info(f"Downloading source code: s3://{_bucket}/{_key} → {_tmp}")
    boto3.client("s3").download_file(_bucket, _key, _tmp)
    os.makedirs(_code_dir, exist_ok=True)
    with tarfile.open(_tmp) as _tar:
        _tar.extractall(_code_dir)
    logger.info(f"Source code extracted to {_code_dir}")

# ---------------------------------------------------------------------------
# Load the inference handler module at import time (before gunicorn forks).
# ---------------------------------------------------------------------------
_program = os.environ.get("SAGEMAKER_PROGRAM", "inference.py")
_code_dir = os.environ.get("PYTHONPATH", "/opt/ml/code").split(":")[0]
_module_path = os.path.join(_code_dir, _program)

logger.info(f"Loading inference handler: {_module_path}")
_spec = importlib.util.spec_from_file_location("_handler", _module_path)
_handler = importlib.util.module_from_spec(_spec)
sys.modules["_handler"] = _handler
_spec.loader.exec_module(_handler)

_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
logger.info(f"Calling model_fn(model_dir={_model_dir!r})")
_model = _handler.model_fn(_model_dir)
logger.info("model_fn complete — server ready")

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return Response("", status=200)


@app.route("/invocations", methods=["POST"])
def invocations():
    content_type = request.content_type or "application/json"
    accept = request.headers.get("Accept", "application/json")
    data = _handler.input_fn(request.get_data(as_text=True), content_type)
    prediction = _handler.predict_fn(data, _model)
    result = _handler.output_fn(prediction, accept)
    return Response(result, mimetype=accept)
