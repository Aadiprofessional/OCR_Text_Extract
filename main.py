import os
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import tempfile
import logging
import shutil
from pathlib import Path

# Try to import PPStructureV3
try:
    from paddleocr import PPStructureV3
    PP_STRUCTURE_AVAILABLE = True
except ImportError:
    PP_STRUCTURE_AVAILABLE = False
    print("WARNING: PPStructureV3 not found in paddleocr. Please ensure paddleocr>=2.8.0 is installed.")

# Initialize FastAPI app
app = FastAPI(title="PaddleOCR API", description="API to extract text/tables using PP-StructureV3")

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PPStructureV3
pipeline = None
if PP_STRUCTURE_AVAILABLE:
    try:
        # Set environment variable to bypass model source check as suggested in logs
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        
        # Initialize with default settings
        # changed use_gpu=False to device='cpu' based on error "Unknown argument: use_gpu"
        # and user's snippet showing device="gpu" is the correct arg.
        pipeline = PPStructureV3(device='cpu') 
        logger.info("PPStructureV3 initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PPStructureV3: {e}")
else:
    logger.error("PPStructureV3 class not available.")

class PDFRequest(BaseModel):
    url: str

def cleanup_temp_file(path: str):
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            logger.info(f"Cleaned up temp path: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up path {path}: {e}")

@app.post("/extract-text")
async def extract_text(request: PDFRequest, background_tasks: BackgroundTasks):
    if not pipeline:
        raise HTTPException(status_code=500, detail="PPStructureV3 is not initialized")

    url = request.url
    logger.info(f"Received request for URL: {url}")

    temp_pdf_path = None
    output_dir = None
    
    try:
        # Download PDF
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF from URL")
        
        # Create a temp file for the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            for chunk in response.iter_content(chunk_size=8192):
                temp_pdf.write(chunk)
            temp_pdf_path = temp_pdf.name
        
        logger.info(f"PDF downloaded to {temp_pdf_path}")

        # Create a temp directory for output
        output_dir = tempfile.mkdtemp()
        output_path = Path(output_dir)

        # Run PPStructureV3
        # According to snippet: output = pipeline.predict(input=input_file)
        # Note: predict() might return a generator or list
        logger.info("Starting PPStructureV3 prediction...")
        output = pipeline.predict(input=temp_pdf_path)
        
        markdown_list = []
        # We don't necessarily need to save images unless we want to return them.
        # For API, we usually return text/markdown.
        
        for res in output:
            # res.markdown is a dict with content
            md_info = res.markdown
            markdown_list.append(md_info)
            
            # If we wanted to save images, we would do it here
            # But we just want the text/markdown content for the API response
        
        # Concatenate markdown pages
        # Check if pipeline has this method (from snippet)
        if hasattr(pipeline, 'concatenate_markdown_pages'):
            markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
        else:
            # Fallback if method doesn't exist
            markdown_texts = "\n\n".join([m.get('markdown', '') for m in markdown_list])

        logger.info("Prediction completed.")

        return {
            "status": "success",
            "markdown": markdown_texts,
            "details": markdown_list # structured data
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Print full stack trace for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Schedule cleanup
        if temp_pdf_path:
            background_tasks.add_task(cleanup_temp_file, temp_pdf_path)
        if output_dir:
            background_tasks.add_task(cleanup_temp_file, output_dir)

# Alias for table extraction (since PPStructureV3 handles both)
@app.post("/extract-table-text")
async def extract_table_text(request: PDFRequest, background_tasks: BackgroundTasks):
    return await extract_text(request, background_tasks)

@app.get("/health")
def health_check():
    status = "ok" if pipeline else "error"
    return {"status": status, "pp_structure_available": PP_STRUCTURE_AVAILABLE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
