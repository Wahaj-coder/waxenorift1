FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    gcc \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir gdown

# Download ALL model weights (OFFLINE, FAIL FAST)
RUN set -eux; \
    mkdir -p /workspace/models; \
    \
    echo "Downloading cricket_ball_detector.pt"; \
    gdown --id 1RFR7QNG0KS8u68IiB4ZR4fZAvyRwxyZ7 \
        -O /workspace/models/cricket_ball_detector.pt; \
    \
    echo "Downloading bestBat.pt"; \
    gdown --id 1MQR-tOl86pAWfhtUtg7PDDDmsTq0eUM1 \
        -O /workspace/models/bestBat.pt; \
    \
    echo "Downloading vitpose-b-multi-coco.pth"; \
    gdown --id 1mHoFS6PEGGx3E0INBdSfFyUr5kUtOUNs \
        -O /workspace/models/vitpose-b-multi-coco.pth; \
    \
    echo "Downloading thirdlstm_shot_classifierupdated.keras"; \
    gdown --id 1G_tJzRtSKaTJmoet0Cma8dCjgJCifTMu \
        -O /workspace/models/thirdlstm_shot_classifierupdated.keras; \
    \
    echo "Downloading 1.csv"; \
    gdown --id 1aKrG286A-JQecHA2IhIuR03fVxd-yMsx \
        -O /workspace/models/1.csv; \
    \
    echo "Downloading cricket_t5_final_clean.zip"; \
    gdown --id 1XheZOO2UO4ZVtupBSNXQwaT09-S-WWtB \
        -O /workspace/models/cricket_t5_final_clean.zip; \
    \
    echo "Unzipping cricket_t5_final_clean.zip"; \
    unzip /workspace/models/cricket_t5_final_clean.zip \
        -d /workspace/models/cricket_t5_final_clean; \
    rm /workspace/models/cricket_t5_final_clean.zip; \
    \
    echo "Downloading YOLOv8n.pt"; \
    gdown --id 19pOyZ3K7zKXUaTAE2TFFmf5Ze9eqnfbc \
        -O /workspace/models/yolov8n.pt; \
    \
    echo "All models downloaded successfully"

RUN apt-get update && apt-get install -y libgeos-dev

COPY requirements.txt .

RUN python -m ensurepip --upgrade

RUN python -m pip install --upgrade setuptools

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/handler.py"]
