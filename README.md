# PaddleOCR API

This is a FastAPI-based API that accepts a PDF URL, extracts text using PaddleOCR, and returns the result.

## Prerequisites

- Docker
- Coolify (or any other Docker-based deployment platform)

## Local Development

1. **Install Dependencies**:
   Note: PaddlePaddle requires specific python versions (3.8-3.12 recommended).
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.

## Deployment to Coolify

This project is ready to be deployed to Coolify using the provided `Dockerfile`.

### Steps:

1.  **Push to Git Repository**:
    - Push this code to a GitHub/GitLab/Bitbucket repository.

2.  **Create Service in Coolify**:
    - Go to your Coolify dashboard.
    - Click **"+ New"** -> **"Project"** -> Select your project -> **"New Resource"**.
    - Select **"Application"** -> **"Public Repository"** (or Private if your repo is private).
    - Enter your repository URL (e.g., `https://github.com/username/repo`).
    - Select the branch (usually `main` or `master`).

3.  **Configuration**:
    - **Build Pack**: Select **Docker Compose** or **Dockerfile**. Since we have a `Dockerfile`, Coolify should automatically detect it.
    - **Port**: Ensure the internal port is set to `8000`.
    - **Domains**: Configure your domain (e.g., `https://ocr-api.yourdomain.com`).

4.  **Deploy**:
    - Click **"Deploy"**.
    - Coolify will build the Docker image and start the container.
    - Watch the build logs to ensure `paddlepaddle` and models are downloaded correctly.

### Environment Variables

No special environment variables are required for basic usage.

## API Usage

### Endpoint: `/extract-text`

**Method**: `POST`

**Body**:
```json
{
  "url": "https://example.com/sample.pdf"
}
```

**Response**:
```json
{
  "status": "success",
  "data": [
    {
      "page": 1,
      "text": "Extracted text content..."
    }
  ]
}
```
# OCR_Text_Extract
