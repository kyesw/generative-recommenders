"""
Minimal SageMaker real-time inference server (FastAPI + uvicorn).

Single process, no forking — CUDA-safe by design.

Routes:
  GET  /ping         → 200 OK  (SageMaker health check)
  POST /invocations  → prediction result

The handler module (SAGEMAKER_PROGRAM) must expose:
    model_fn(model_dir)                    → model context
    input_fn(request_body, content_type)   → parsed input
    predict_fn(data, model)                → prediction
    output_fn(prediction, accept)          → serialised response string
"""

import importlib.util
import logging
import os
import sys

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load the inference handler module at startup.
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
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI()


@app.get("/ping")
def ping():
    return PlainTextResponse("", status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    content_type = request.headers.get("content-type", "application/json")
    accept = request.headers.get("accept", "application/json")
    # Normalize wildcard accept to application/json
    if accept == "*/*" or not accept:
        accept = "application/json"
    body = (await request.body()).decode("utf-8")
    data = _handler.input_fn(body, content_type)
    prediction = _handler.predict_fn(data, _model)
    result = _handler.output_fn(prediction, accept)
    return Response(content=result, media_type="application/json")
