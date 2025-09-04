# ======================================================================
#  FusionNetGeoLabel Dockerfile
# ======================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git wget unzip curl gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/FusionNetGeoLabel
COPY . .

RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: set default to train with mounted data
CMD ["python3", "train.py", "--config", "configs/default.yaml"]
