FROM python:3.9-bullseye

# Install system dependencies
# Using bullseye which is more stable than slim
# Added --allow-releaseinfo-change to handle potential repository changes
RUN set -eux; \
    apt-get update --allow-releaseinfo-change; \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        curl \
        build-essential \
        python3-dev; \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install python dependencies
# Using --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for PaddleOCR
ENV PADDLEOCR_CACHE_DIR=/tmp/paddleocr_cache
ENV PADDLEX_HOME=/tmp/paddlex
ENV PADDLE_PDX_CACHE_HOME=/tmp/paddlex_home
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Run the model download script
RUN python download_models.py || echo "Model download failed, but continuing build. Models will be downloaded at runtime."

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
