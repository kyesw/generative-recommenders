"""
SageMaker entry point for the research pipeline (HSTU/SASRec via main.py).

SageMaker passes hyperparameters as a JSON file at:
  /opt/ml/input/config/hyperparameters.json

Supported hyperparameters:
  gin_config_file  : path to gin config (default: configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin)
  dataset_name     : dataset key understood by get_common_preprocessors() (default: ml-1m)
  master_port      : DDP master port (default: 12345)
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

gin_config_file: str = hyperparameters.get(
    "gin_config_file",
    "configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin",
)
dataset_name: str = hyperparameters.get("dataset_name", "ml-1m")
master_port: str = str(hyperparameters.get("master_port", "12345"))

# ---------------------------------------------------------------------------
# 2. Point output to /opt/ml/model so SageMaker uploads it to S3
# ---------------------------------------------------------------------------
output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.environ["OUTPUT_DIR"] = output_dir
logger.info(f"OUTPUT_DIR set to: {output_dir}")

# ---------------------------------------------------------------------------
# 3. Create tmp/ for data downloads (relative to CWD = /opt/ml/code)
# ---------------------------------------------------------------------------
os.makedirs("tmp", exist_ok=True)

# ---------------------------------------------------------------------------
# 4. Download and preprocess data
# ---------------------------------------------------------------------------
logger.info(f"Preprocessing dataset: {dataset_name}")
from generative_recommenders.research.data.preprocessor import get_common_preprocessors

preprocessors = get_common_preprocessors()
if dataset_name not in preprocessors:
    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        f"Supported: {list(preprocessors.keys())}"
    )
preprocessors[dataset_name].preprocess_rating()
logger.info("Data preprocessing complete.")

# ---------------------------------------------------------------------------
# 5. Delegate to main.main() with the correct flags
# ---------------------------------------------------------------------------
sys.argv = [
    "train_research.py",
    f"--gin_config_file={gin_config_file}",
    f"--master_port={master_port}",
]
logger.info(f"Launching training with argv: {sys.argv}")

import main  # noqa: E402  (repo root is on PYTHONPATH via /opt/ml/code)

main.main()
