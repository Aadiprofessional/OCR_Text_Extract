FROM python:3.9-slim

# Install system dependencies
# We are encountering issues with apt-get update on the default mirrors.
# Attempting to use a more robust update strategy:
# 1. Update package lists
# 2. Install dependencies
# 3. Clean up
RUN set -eux; \
    apt-get update; \
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
RUN pip install --no-cache-dir -r requirements.txt

# Download paddleocr models during build to avoid downloading them at runtime
# We can do this by running a simple python script that initializes PaddleOCR
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
