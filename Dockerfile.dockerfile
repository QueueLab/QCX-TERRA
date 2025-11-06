# Dockerfile
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Copy code + env
COPY . /app
WORKDIR /app

# Create conda env
COPY conda.yaml .
RUN conda env create -f conda.yaml && conda clean -afy

# Activate env for runtime
ENV PATH="/opt/conda/envs/hf-onnx-env/bin:$PATH"

# Azure ML entry point
ENTRYPOINT ["python", "score.py"]