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

from flask import Flask, Response, request

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

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
