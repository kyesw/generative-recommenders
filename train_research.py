"""
SageMaker entry point for the research pipeline (HSTU/SASRec via main.py).

SageMaker passes hyperparameters as a JSON file at:
  /opt/ml/input/config/hyperparameters.json

Supported hyperparameters:
  gin_config_file  : path to gin config (default: configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin)
  dataset_name     : dataset identifier (default: ml-1m)
  master_port      : DDP master port (default: 12345)
  
  Tunable hyperparameters (override gin config):
  learning_rate    : learning rate (float)
  local_batch_size : batch size (int)
  dropout_rate     : dropout rate (float)
  num_negatives    : number of negative samples (int)
  temperature      : softmax temperature (float)

Data loading (in priority order):
  1. S3 input channel  — when launched with --data-s3-uri via launch_training.py,
                         SageMaker downloads the data to SM_CHANNEL_TRAINING before
                         this script runs. The CSV is symlinked into tmp/ so the
                         rest of the pipeline finds it at the expected path.
  2. Download at runtime — fallback when no input channel is provided.
                           Downloads and preprocesses from the public URL.
"""

import json
import logging
import os
import sys
import tempfile

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Guard: when mp.spawn/forkserver re-imports this module in child processes,
# __name__ is set to '__mp_main__', not '__main__'. Without this guard the
# entire setup block (including main.main()) would re-execute in every child,
# causing "context has already been set" errors.
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 1. Read SageMaker hyperparameters
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 2. Point output to /opt/ml/model so SageMaker uploads it to S3
    # -----------------------------------------------------------------------
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.environ["OUTPUT_DIR"] = output_dir
    logger.info(f"OUTPUT_DIR set to: {output_dir}")

    # -----------------------------------------------------------------------
    # 3. Make training data available under tmp/
    #    Priority: S3 input channel > download at runtime
    #
    #    prepare_data.py uploads the full /tmp/ tree to S3, preserving paths:
    #      s3://.../ml-1m/ml-1m/sasrec_format.csv
    #      s3://.../ml-1m/processed/ml-1m/movies.csv  ...
    #    SageMaker downloads that to SM_CHANNEL_TRAINING, so we symlink
    #    tmp/ -> SM_CHANNEL_TRAINING and the code finds all files as expected.
    # -----------------------------------------------------------------------
    training_channel = os.environ.get("SM_CHANNEL_TRAINING", "")

    if training_channel:
        # --- Option 2: full tmp/ tree was downloaded from S3 by SageMaker ---
        logger.info(f"SM_CHANNEL_TRAINING contents ({training_channel}):")
        for root, dirs, files in os.walk(training_channel):
            for f in files:
                logger.info(f"  {os.path.join(root, f)}")

        if os.path.islink("tmp"):
            os.unlink("tmp")
        elif os.path.isdir("tmp"):
            import shutil
            shutil.rmtree("tmp")
        os.symlink(training_channel, "tmp")
        logger.info(f"Symlinked tmp/ -> {training_channel}")
    else:
        # --- Option 1: download and preprocess at runtime ---
        logger.info(f"No S3 input channel found. Downloading dataset: {dataset_name}")
        os.makedirs(f"tmp/{dataset_name}", exist_ok=True)
        from generative_recommenders.research.data.preprocessor import (
            get_common_preprocessors,
        )

        preprocessors = get_common_preprocessors()
        if dataset_name not in preprocessors:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {list(preprocessors.keys())}"
            )
        preprocessors[dataset_name].preprocess_rating()
        logger.info("Data preprocessing complete.")

    # -----------------------------------------------------------------------
    # 4. Create modified gin config with hyperparameter overrides
    # -----------------------------------------------------------------------
    tunable_params = {
        "learning_rate": ("train_fn.learning_rate", float),
        "local_batch_size": ("train_fn.local_batch_size", int),
        "dropout_rate": ("train_fn.dropout_rate", float),
        "num_negatives": ("train_fn.num_negatives", int),
        "temperature": ("train_fn.temperature", float),
    }
    
    # Check if we have any overrides
    has_overrides = any(param_name in hyperparameters for param_name in tunable_params.keys())
    
    if has_overrides:
        # Read original gin config
        with open(gin_config_file, 'r') as f:
            original_config = f.read()
        
        # Create a temporary gin config with overrides appended
        override_lines = ["\n# Hyperparameter overrides from SageMaker tuner:"]
        for param_name, (gin_param, param_type) in tunable_params.items():
            if param_name in hyperparameters:
                value = param_type(hyperparameters[param_name])
                override_lines.append(f"{gin_param} = {value}")
                logger.info(f"Override: {gin_param} = {value}")
        
        # Write to temporary file
        temp_gin = tempfile.NamedTemporaryFile(mode='w', suffix='.gin', delete=False, dir='/tmp')
        temp_gin.write(original_config)
        temp_gin.write('\n'.join(override_lines))
        temp_gin.close()
        
        gin_config_file = temp_gin.name
        logger.info(f"Created temporary gin config with overrides: {gin_config_file}")
    else:
        logger.info("No hyperparameter overrides - using gin config defaults")

    # -----------------------------------------------------------------------
    # 5. Delegate to main.main() with the correct flags
    # -----------------------------------------------------------------------
    sys.argv = [
        "train_research.py",
        f"--gin_config_file={gin_config_file}",
        f"--master_port={master_port}",
    ]
    logger.info(f"Launching training with argv: {sys.argv}")

    import main  # noqa: E402  (repo root is on PYTHONPATH via /opt/ml/code)

    main.main()
