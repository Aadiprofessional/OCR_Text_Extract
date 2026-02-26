# PaddleOCR API with PP-StructureV3

This API uses **PP-StructureV3** (via PaddleOCR) to extract text and tables from PDF documents. It handles complex layouts and tables automatically and returns the content in Markdown format.

## Features

- **End-to-End Extraction**: Uses `PP-StructureV3` exclusively for document understanding.
- **Table Support**: Automatically detects and extracts tables into Markdown tables.
- **PDF Support**: Downloads and processes multi-page PDFs directly.
- **Markdown Output**: Returns the full document content as a concatenated Markdown string.

## Prerequisites

- Python 3.8 - 3.10 (Recommended for PaddlePaddle)
- Docker (optional but recommended for consistent environment)

## Local Development

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you are on macOS with Apple Silicon (M1/M2), ensure you install the correct `paddlepaddle` wheel or use Docker.*

2.  **Run the API**:
    ```bash
    python main.py
    ```
    The API will be available at `http://localhost:8000`.

## Deployment (Docker / Coolify)

1.  **Build and Run with Docker**:
    ```bash
    docker-compose up --build
    ```

2.  **Deploy to Coolify**:
    - Connect your repository.
    - Use **Docker Compose** build pack.
    - Deploy.

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
  "markdown": "# Document Title\n\nSome text content...\n\n| Table | Header |\n|-------|--------|\n| Cell  | Cell   |\n\n...",
  "details": [ ...structured data per page... ]
}
```

### Endpoint: `/extract-table-text`

Alias for `/extract-text`.
