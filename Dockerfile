FROM python:3.11-slim 
 
# Install system dependencies 
RUN apt-get update && apt-get install -y \ 
    poppler-utils \ 
    libglib2.0-0 \ 
    libsm6 \ 
    libxext6 \ 
    libxrender-dev \ 
    libgomp1 \ 
    wget \ 
    && rm -rf /var/lib/apt/lists/* 
 
WORKDIR /app 
 
COPY requirements.txt . 
 
# Install Python deps 
RUN pip install --no-cache-dir -r requirements.txt 
 
# Pre-download PP-StructureV3 models at build time (so first request is fast) 
RUN python -c "from paddleocr import PPStructureV3; PPStructureV3()" 
 
COPY main.py . 
 
EXPOSE 8000 
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]