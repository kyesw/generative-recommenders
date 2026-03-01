# SageMaker PyTorch 2.5 with CUDA 12.4 (closest to 2.6)
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install fbgemm-gpu and torchrec from PyTorch CUDA 12.4 wheel index
RUN pip install --no-cache-dir \
    fbgemm-gpu \
    torchrec \
    --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
RUN pip install --no-cache-dir \
    gin-config \
    pandas \
    pybind11 \
    click \
    matplotlib \
    absl-py \
    mlflow \
    sagemaker-mlflow

# Copy repo and compile CUDA extensions in place
COPY . /opt/ml/code/
WORKDIR /opt/ml/code
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/opt/ml/code
# Default entry point; overridable via SAGEMAKER_PROGRAM env var
ENV SAGEMAKER_PROGRAM=train_research.py
