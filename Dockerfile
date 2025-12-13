# ============================================================
# BASE IMAGE — Python 3.10 slim (GPU-ready via CUDA wheels)
# ============================================================
FROM python:3.10-slim

# ------------------------------------------------------------
# Set working directory inside container
# ------------------------------------------------------------
WORKDIR /workspace

# ------------------------------------------------------------
# Install system-level dependencies
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Install gdown FIRST (for Google Drive downloads)
# ------------------------------------------------------------
RUN pip install --no-cache-dir gdown

# ------------------------------------------------------------
# Download model weights from Google Drive (FIRST, FAIL FAST)
# ------------------------------------------------------------
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
    \
    echo "Listing contents of /workspace/models/cricket_t5_final_clean:"; \
    ls -l /workspace/models/cricket_t5_final_clean; \
    \
    echo "Removing zip file"; \
    rm /workspace/models/cricket_t5_final_clean.zip; \
    echo "Zip file removed successfully"

# ------------------------------------------------------------
# Install GPU-enabled PyTorch & TorchVision (CUDA 12.1)
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# Install the rest of your Python dependencies
# ------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Clone ViTPose repository and install dependencies
# ------------------------------------------------------------
RUN git clone https://github.com/jaehyunnn/ViTPose_pytorch.git /workspace/ViTPose_pytorch

# ------------------------------------------------------------
# Copy app.py into container
# ------------------------------------------------------------
COPY app.py .

# ------------------------------------------------------------
# Expose API port
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# Start Gunicorn with 2 workers (tune -w based on VRAM)
# ------------------------------------------------------------
CMD ["gunicorn", "-w", "2", "--threads", "2", "-b", "0.0.0.0:8000", "app:app"]
