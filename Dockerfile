FROM python:3.9-bullseye

# Install system dependencies
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
RUN pip install --no-cache-dir -r requirements.txt

# Download paddleocr models during build (optional but good practice)
# We try to initialize PPStructureV3 to trigger downloads
# Note: If PPStructureV3 is not available in the installed version, this might fail build.
# We will use a try-except block in the build script or just rely on runtime download.
# For now, let's just copy the code.

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
