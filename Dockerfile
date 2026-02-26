FROM python:3.9-bullseye

# Install system dependencies
# Using bullseye which is more stable than slim
# Added --allow-releaseinfo-change to handle potential repository changes
RUN set -eux; \
    apt-get update --allow-releaseinfo-change; \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        curl; \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install python dependencies
# Using --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install opencv-python-headless

# Copy model download script
COPY download_models.py .

# Run model download script
# This script is designed to exit gracefully (code 0) even if download fails,
# preventing build failure. Models will be downloaded at runtime if needed.
RUN python download_models.py

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
