import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HOME", PROJECT_DIR)
os.environ.setdefault("PADDLEX_HOME", os.path.join(PROJECT_DIR, ".paddlex"))
os.environ.setdefault("PADDLEX_CACHE_DIR", os.path.join(PROJECT_DIR, ".paddlex", "temp"))
os.environ.setdefault("PADDLE_HUB_HOME", os.path.join(PROJECT_DIR, ".paddlex", "official_models"))
os.environ.setdefault("HUB_HOME", os.path.join(PROJECT_DIR, ".paddlex", "official_models"))
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import tempfile
import logging
from paddleocr import PPStructureV3

app = FastAPI(title="PP-StructureV3 API", description="API using PP-StructureV3 for OCR and table extraction")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    pipeline = PPStructureV3(
        lang="en",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True
    )
    logger.info("PP-StructureV3 initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PP-StructureV3: {e}")
    raise e

class PDFRequest(BaseModel):
    url: str

def cleanup_temp_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Cleaned up temp file: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {path}: {e}")

@app.post("/extract-text")
async def extract_text(request: PDFRequest, background_tasks: BackgroundTasks):
    url = request.url.strip()
    logger.info(f"Received request for URL: {url}")
    temp_pdf_path = None
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download file from URL")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            for chunk in response.iter_content(chunk_size=8192):
                temp_pdf.write(chunk)
            temp_pdf_path = temp_pdf.name
        output = pipeline.predict(input=temp_pdf_path)
        markdown_list = []
        pages = []
        for idx, res in enumerate(output):
            md_info = res.markdown
            markdown_list.append(md_info)
            pages.append({"page": idx + 1, "markdown": md_info})
        combined_markdown = pipeline.concatenate_markdown_pages(markdown_list)
        return {"status": "success", "data": pages, "combined_markdown": combined_markdown}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_pdf_path:
            background_tasks.add_task(cleanup_temp_file, temp_pdf_path)

@app.post("/extract-table-text")
async def extract_table_text(request: PDFRequest, background_tasks: BackgroundTasks):
    url = request.url.strip()
    logger.info(f"Received request for URL: {url} (extract-table-text)")
    temp_pdf_path = None
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download file from URL")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            for chunk in response.iter_content(chunk_size=8192):
                temp_pdf.write(chunk)
            temp_pdf_path = temp_pdf.name
        output = pipeline.predict(input=temp_pdf_path)
        markdown_list = []
        pages = []
        for idx, res in enumerate(output):
            md_info = res.markdown
            markdown_list.append(md_info)
            pages.append({"page": idx + 1, "markdown": md_info})
        combined_markdown = pipeline.concatenate_markdown_pages(markdown_list)
        return {"status": "success", "data": pages, "combined_markdown": combined_markdown}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_pdf_path:
            background_tasks.add_task(cleanup_temp_file, temp_pdf_path)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
