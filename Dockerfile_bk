# Use official PyTorch 2.6.0 with CUDA 12.4 (matches requirements.txt)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install SageMaker training toolkit (needed for SageMaker integration)
RUN pip install --no-cache-dir sagemaker-training sagemaker-inference

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libgomp1 \
    nginx \
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
    sagemaker-mlflow \
    flask \
    gunicorn \
    gevent

# Copy minimal files for package installation
WORKDIR /opt/ml/code
COPY setup.py requirements.txt ./
COPY generative_recommenders generative_recommenders/

# Install the package (this compiles CUDA extensions)
RUN pip install --no-cache-dir -e .

# Create entrypoint script that handles both train and serve
RUN echo '#!/bin/bash\n\
if [ "$1" = "train" ]; then\n\
    exec train\n\
elif [ "$1" = "serve" ]; then\n\
    exec serve\n\
else\n\
    exec "$@"\n\
fi' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENV PYTHONPATH=/opt/ml/code
ENV SAGEMAKER_PROGRAM=train_research.py

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["train"]

# Note: Full code directory will be uploaded by SageMaker at runtime via source_dir
# This image only needs dependencies + compiled CUDA extensions
