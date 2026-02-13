import os
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from paddleocr import PaddleOCR
import tempfile
import logging

# Initialize FastAPI app
app = FastAPI(title="PaddleOCR API", description="API to extract text from PDF URLs using PaddleOCR")

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PaddleOCR (load model once)
# use_angle_cls=True enables angle classification
# lang='en' sets the language to English. Change if needed.
# enable_mkldnn=False to avoid "ConvertPirAttribute2RuntimeAttribute" errors in some environments
# Note: newer versions of paddleocr might have changed argument names. 
# 'use_gpu' is not a valid argument for the new Pipeline-based API in some versions.
# We will try to set it via environment variable if needed, or rely on defaults.
try:
    # Removed use_gpu=False as it caused ValueError: Unknown argument: use_gpu
    ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    logger.info("PaddleOCR initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR: {e}")
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
    url = request.url
    logger.info(f"Received request for URL: {url}")

    temp_pdf_path = None
    
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

        # Open PDF with PyMuPDF
        doc = fitz.open(temp_pdf_path)
        extracted_text = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            
            # Create a temp image file for the page
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                pix.save(temp_img.name)
                temp_img_path = temp_img.name
            
            try:
                # Run OCR on the image
                # cls=True is removed from the call as it might be causing issues with the current version.
                # The initialization PaddleOCR(use_angle_cls=True) should be sufficient.
                result = ocr.ocr(temp_img_path)
                
                # Log the raw result structure for debugging
                logger.info(f"Page {page_num + 1} raw result: {result}")

                page_text = ""
                if result:
                    # The latest PaddleOCR output format seems to be a list of dictionaries (or a single dictionary in a list)
                    # Based on logs: [{'rec_texts': ['PRINCE', ...], 'rec_scores': [...], ...}]
                    
                    # Handle list of dicts (new format)
                    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                        for res in result:
                            if 'rec_texts' in res:
                                # New format: text is in 'rec_texts' list
                                texts = res.get('rec_texts', [])
                                page_text += "\n".join(texts) + "\n"
                            else:
                                # Fallback or unknown format, try to log keys
                                logger.warning(f"Unknown dict keys in result: {res.keys()}")
                    
                    # Handle legacy format (list of lists)
                    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                         ocr_result = result[0]
                         for line in ocr_result:
                            if line and len(line) > 1:
                                try:
                                    text = line[1][0]
                                    page_text += text + "\n"
                                except (IndexError, TypeError):
                                    pass
                    
                    # Handle unexpected format
                    else:
                         logger.warning(f"Unexpected result format: {type(result)}")

                extracted_text.append({
                    "page": page_num + 1,
                    "text": page_text.strip()
                })
            finally:
                # Clean up image file immediately
                cleanup_temp_file(temp_img_path)

        return {"status": "success", "data": extracted_text}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Schedule PDF file cleanup
        if temp_pdf_path:
            background_tasks.add_task(cleanup_temp_file, temp_pdf_path)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
