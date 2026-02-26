FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install paddlepaddle 3.0.0 from official Paddle index (required for PPStructureV3)
RUN pip install --no-cache-dir paddlepaddle==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Step 2: Install paddleocr 3.0.1 (last stable version without the PaddlePredictorOption bug)
RUN pip install --no-cache-dir "paddleocr[all]==3.0.1"

# Step 3: Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]