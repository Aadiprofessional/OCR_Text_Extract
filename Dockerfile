FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install paddlepaddle CPU first, then paddleocr
RUN pip install --no-cache-dir paddlepaddle==3.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir "paddleocr[all]==3.0.3"
RUN pip install --no-cache-dir fastapi uvicorn pdf2image pillow requests python-multipart

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]