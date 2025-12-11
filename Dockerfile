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
# Create required folders
# ------------------------------------------------------------
RUN mkdir -p /workspace/models \
    /workspace/processed \
    /workspace/ViTPose_pytorch

# ------------------------------------------------------------
# Copy requirements first for layer caching
# (requirements.txt should NOT include torch / tensorflow)
# ------------------------------------------------------------
COPY requirements.txt .

# ------------------------------------------------------------
# Upgrade pip
# ------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip

# ------------------------------------------------------------
# Install GPU-enabled PyTorch & TorchVision (CUDA 12.1)
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# Install GPU-enabled TensorFlow
# ------------------------------------------------------------
RUN pip install --no-cache-dir "tensorflow[and-cuda]==2.17.0"

# ------------------------------------------------------------
# Install the rest of your Python dependencies
# (torch / tf already installed as GPU builds)
# ------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Clone ViTPose repository and install its dependencies
# ------------------------------------------------------------
RUN git clone https://github.com/jaehyunnn/ViTPose_pytorch.git /workspace/ViTPose_pytorch && \
    pip install --no-cache-dir -r /workspace/ViTPose_pytorch/requirements.txt

# ------------------------------------------------------------
# Download model weights from Google Drive (BAKED INTO IMAGE)
# Uses confirm-token trick to bypass virus scan page
# ------------------------------------------------------------
RUN set -eux; \
    mkdir -p /workspace/models; \
    download_from_gdrive() { \
        fileid="$1"; \
        filename="$2"; \
        tmpdir="$(mktemp -d)"; \
        cd "$tmpdir"; \
        wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate \
          "https://docs.google.com/uc?export=download&id=${fileid}" -O- \
          | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt || true; \
        confirm="$(cat confirm.txt 2>/dev/null || true)"; \
        if [ -n "$confirm" ]; then \
          wget --quiet --load-cookies cookies.txt --no-check-certificate \
            "https://docs.google.com/uc?export=download&confirm=${confirm}&id=${fileid}" \
            -O "$filename"; \
        else \
          wget --quiet --load-cookies cookies.txt --no-check-certificate \
            "https://docs.google.com/uc?export=download&id=${fileid}" \
            -O "$filename"; \
        fi; \
        mv "$filename" /workspace/models/"$filename"; \
        cd /workspace; \
        rm -rf "$tmpdir"; \
    }; \
    \
    # 1) Ball YOLO model
    download_from_gdrive "1RFR7QNG0KS8u68IiB4ZR4fZAvyRwxyZ7" "cricket_ball_detector.pt"; \
    \
    # 2) Bat YOLO model
    download_from_gdrive "1MQR-tOl86pAWfhtUtg7PDDDmsTq0eUM1" "bestBat.pt"; \
    \
    # 3) ViTPose checkpoint
    download_from_gdrive "1mHoFS6PEGGx3E0INBdSfFyUr5kUtOUNs" "vitpose-b-multi-coco.pth"; \
    \
    # 4) LSTM Keras model
    download_from_gdrive "1G_tJzRtSKaTJmoet0Cma8dCjgJCifTMu" "thirdlstm_shot_classifierupdated.keras"; \
    \
    # 5) Reference CSV
    download_from_gdrive "1aKrG286A-JQecHA2IhIuR03fVxd-yMsx" "1.csv"; \
    \
    # 6) T5 model folder (Drive folder -> downloaded as zip)
    download_from_gdrive "1n3IDbXWGjbm0dfxuM7y5cy_WDx1FwagS" "cricket_t5_final_clean.zip"; \
    unzip /workspace/models/cricket_t5_final_clean.zip -d /workspace/models/cricket_t5_final_clean; \
    rm /workspace/models/cricket_t5_final_clean.zip

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
