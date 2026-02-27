"""
SageMaker entry point for the DLRM v3 production pipeline.

SageMaker passes hyperparameters as a JSON file at:
  /opt/ml/input/config/hyperparameters.json

Supported hyperparameters:
  dataset  : dataset key recognised by train_ranker.SUPPORTED_CONFIGS (default: debug)
  mode     : train | eval | train-eval | streaming-train-eval (default: train)
"""

import json
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Read SageMaker hyperparameters
# ---------------------------------------------------------------------------
HP_PATH = "/opt/ml/input/config/hyperparameters.json"
hyperparameters: dict = {}
if os.path.isfile(HP_PATH):
    with open(HP_PATH) as f:
        hyperparameters = json.load(f)
    logger.info(f"Loaded hyperparameters: {hyperparameters}")
else:
    logger.warning(f"No hyperparameters file found at {HP_PATH}; using defaults.")

dataset: str = hyperparameters.get("dataset", "debug")
mode: str = hyperparameters.get("mode", "train")

# ---------------------------------------------------------------------------
# 2. Set output and distributed env vars
# ---------------------------------------------------------------------------
output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.environ["SM_MODEL_DIR"] = output_dir
os.environ["OUTPUT_DIR"] = output_dir
logger.info(f"OUTPUT_DIR set to: {output_dir}")

# Single-GPU defaults; SageMaker sets these automatically for multi-instance,
# but we ensure they are present for single-GPU runs.
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("WORLD_SIZE", "1")

# ---------------------------------------------------------------------------
# 3. Delegate to train_ranker.main() with reconstructed argv
# ---------------------------------------------------------------------------
sys.argv = [
    "train_dlrm_v3.py",
    f"--dataset={dataset}",
    f"--mode={mode}",
]
logger.info(f"Launching DLRM v3 training with argv: {sys.argv}")

from generative_recommenders.dlrm_v3.train.train_ranker import main  # noqa: E402

main()
