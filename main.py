import os
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
try:
    from paddleocr import PaddleOCR, PPStructure
except ImportError:
    # Fallback for older versions or if PPStructure is missing
    from paddleocr import PaddleOCR
    PPStructure = None
    logging.warning("PPStructure not found in paddleocr module. Table extraction will be disabled/limited.")

import tempfile
import logging
import numpy as np
import concurrent.futures
import time

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
    
    # Initialize PPStructure for table extraction
    # This might download models on first run
    if PPStructure:
        table_engine = PPStructure(show_log=True, image_orientation=True)
        logger.info("PPStructure initialized successfully.")
    else:
        table_engine = None
        logger.warning("PPStructure not available. Table extraction will fail.")

except Exception as e:
    logger.error(f"Failed to initialize OCR engines: {e}")
    # Don't crash immediately, let specific routes fail if needed, or re-raise
    # But usually better to fail fast if core functionality is broken
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

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Truncate large arrays (likely images) to avoid JSON bloat
        if obj.size > 100:
             return f"<large_array_shape_{obj.shape}>"
        return obj.tolist()
    elif isinstance(obj, list):
        # Recursively check list items, truncate large lists
        if len(obj) > 100 and isinstance(obj[0], (int, float, np.integer, np.floating)):
             return f"<large_list_len_{len(obj)}>"
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, dict):
        # Filter out keys that might contain large data (e.g. 'img', 'pixels')
        clean_dict = {}
        for k, v in obj.items():
            if k in ['img', 'image', 'pixels', 'data']: # Common large keys
                clean_dict[k] = "<truncated_image_data>"
            else:
                clean_dict[k] = convert_numpy_types(v)
        return clean_dict
    else:
        return obj

def process_ocr_result(result):
    processed_result = []
    
    if not result:
        return processed_result

    # Handle list of dicts (new format)
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        for res in result:
            processed_result.append(convert_numpy_types(res))
            
    # Handle legacy format (list of lists)
    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        ocr_result = result[0]
        for line in ocr_result:
            if line and len(line) > 1:
                box = line[0]
                text_info = line[1]
                if len(text_info) >= 2:
                    text = text_info[0]
                    score = text_info[1]
                    processed_result.append({
                        "box": convert_numpy_types(box),
                        "text": text,
                        "score": convert_numpy_types(score)
                    })
    
    return processed_result

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

def process_page_ocr(page_num, temp_img_path):
    try:
        start_time = time.time()
        # Run OCR on the image
        result = ocr.ocr(temp_img_path)
        logger.info(f"Page {page_num} OCR processed in {time.time() - start_time:.2f}s")
        
        # Log the raw result structure for debugging (only if small or just type)
        if result:
             logger.debug(f"Page {page_num} raw result type: {type(result)}")

        page_data = process_ocr_result(result)

        return {
            "page": page_num,
            "result": page_data
        }
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        return {
            "page": page_num,
            "error": str(e)
        }
    finally:
        # Clean up image file immediately after OCR is done
        cleanup_temp_file(temp_img_path)

def process_page_structure(page_num, temp_img_path):
    if not table_engine:
         return {"page": page_num, "error": "PPStructure not initialized"}

    try:
        start_time = time.time()
        # Run Structure Analysis on the image
        # This returns a list of regions (tables, figures, text, etc.)
        # Each region has 'type', 'bbox', 'res' (html for tables, text for others), and 'img' (cropped image)
        result = table_engine(temp_img_path)
        logger.info(f"Page {page_num} Structure Analysis processed in {time.time() - start_time:.2f}s")
        
        # Log result type
        if result:
             logger.debug(f"Page {page_num} structure result type: {type(result)}")

        # Process result to remove large images and convert types
        page_data = []
        if isinstance(result, list):
            for region in result:
                # Remove large image data if present (it's usually in 'img' key)
                # convert_numpy_types handles filtering 'img' key
                clean_region = convert_numpy_types(region)
                page_data.append(clean_region)
        else:
            page_data = convert_numpy_types(result)

        return {
            "page": page_num,
            "result": page_data
        }
    except Exception as e:
        logger.error(f"Error processing structure for page {page_num}: {e}")
        return {
            "page": page_num,
            "error": str(e)
        }
    finally:
        # Clean up image file immediately
        cleanup_temp_file(temp_img_path)

@app.post("/extract-table-text")
async def extract_table_text(request: PDFRequest, background_tasks: BackgroundTasks):
    url = request.url.strip()
    logger.info(f"Received request for URL: {url} (extract-table-text)")

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
        
        # Extract all images first sequentially (fast)
        page_tasks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            
            # Create a temp image file for the page
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                pix.save(temp_img.name)
                temp_img_path = temp_img.name
                page_tasks.append((page_num + 1, temp_img_path))
        
        logger.info(f"Extracted {len(page_tasks)} pages as images. Starting parallel Structure Analysis.")
        
        extracted_data = []
        
        # Use ThreadPoolExecutor for parallel processing
        # Structure analysis can be heavy, so limit workers if needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Use process_page_structure instead of process_page_ocr
            future_to_page = {executor.submit(process_page_structure, p_num, p_path): p_num for p_num, p_path in page_tasks}
            
            for future in concurrent.futures.as_completed(future_to_page):
                try:
                    data = future.result()
                    extracted_data.append(data)
                except Exception as exc:
                    logger.error(f"Page processing generated an exception: {exc}")
        
        # Sort results by page number
        extracted_data.sort(key=lambda x: x['page'])

        return {"status": "success", "data": extracted_data}

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
