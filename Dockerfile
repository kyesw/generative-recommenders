FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

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
    tensorboard \
    pybind11 \
    click \
    matplotlib \
    absl-py \
    sagemaker-training

# Copy repo and compile CUDA extensions in place
COPY . /opt/ml/code/
WORKDIR /opt/ml/code
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/opt/ml/code
# Default entry point; overridable via SAGEMAKER_PROGRAM env var
ENV SAGEMAKER_PROGRAM=train_research.py
