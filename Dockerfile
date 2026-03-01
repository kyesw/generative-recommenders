# Use official PyTorch 2.6.0 with CUDA 12.4 (matches requirements.txt)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install SageMaker training and inference toolkits
RUN pip install --no-cache-dir sagemaker-training sagemaker-inference

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install fbgemm-gpu and torchrec from PyTorch CUDA 12.4 wheel index
RUN pip install --no-cache-dir \
    fbgemm-gpu>=1.1.0 \
    torchrec>=1.1.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
RUN pip install --no-cache-dir \
    gin-config>=0.5.0 \
    pandas>=2.2.0 \
    tensorboard>=2.19.0 \
    pybind11 \
    click \
    matplotlib \
    absl-py \
    mlflow \
    sagemaker-mlflow

# Copy minimal files for package installation
WORKDIR /opt/ml/code
COPY setup.py requirements.txt ./
COPY generative_recommenders generative_recommenders/

# Install the package (this compiles CUDA extensions)
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/opt/ml/code

# Note: SAGEMAKER_PROGRAM will be set at runtime by SageMaker
# Training: automatically set to entry_point script
# Inference: set via environment or use inference.py as default

# Note: Full code directory will be uploaded by SageMaker at runtime via source_dir
# This image only needs dependencies + compiled CUDA extensions
