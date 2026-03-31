import os
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import tempfile
import logging
import numpy as np
import concurrent.futures
import time
import cv2
from urllib.parse import urlparse
from typing import Optional
import subprocess
import shutil
import re

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
try:
    from img2table.ocr import PaddleOCR as Img2TablePaddleOCR
    from img2table.document import Image as Img2TableImage
    IMG2TABLE_AVAILABLE = True
except ImportError:
    IMG2TABLE_AVAILABLE = False
    print("WARNING: img2table not found. Advanced table extraction disabled.")

try:
    from paddleocr import PaddleOCR, PPStructure
except ImportError:
    # Fallback for older versions or if PPStructure is missing
    from paddleocr import PaddleOCR
    PPStructure = None
    # Can't use logging here yet as it's not imported/configured
    print("WARNING: PPStructure not found in paddleocr module. Table extraction will be disabled/limited.")

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
        logger.warning("PPStructure not available (ImportError). Table extraction will fail.")

except Exception as e:
    logger.error(f"Failed to initialize OCR engines: {e}")
    # Don't crash immediately, let specific routes fail if needed, or re-raise
    # But usually better to fail fast if core functionality is broken
    raise e

class PDFRequest(BaseModel):
    url: str
    lang: Optional[str] = "auto"

def sanitize_input_url(raw_url: str):
    if not raw_url:
        raise HTTPException(status_code=400, detail="URL is required")
    clean_url = str(raw_url).replace("\n", " ").replace("\r", " ").strip()
    match = re.search(r"https?://[^\s'\"`]+", clean_url)
    if match:
        clean_url = match.group(0).strip()
    clean_url = clean_url.strip("`'\"")
    if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Invalid URL. Use a valid http/https URL")
    parsed = urlparse(clean_url)
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL. Missing host")
    return clean_url

def guess_file_suffix(url: str, content_type: Optional[str]):
    parsed = urlparse(url)
    _, ext = os.path.splitext(parsed.path.lower())
    if ext in [".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        return ext
    if content_type:
        ct = content_type.lower()
        if "pdf" in ct:
            return ".pdf"
        if "vnd.openxmlformats-officedocument.wordprocessingml.document" in ct:
            return ".docx"
        if "application/msword" in ct:
            return ".doc"
        if "png" in ct:
            return ".png"
        if "jpeg" in ct or "jpg" in ct:
            return ".jpg"
        if "bmp" in ct:
            return ".bmp"
        if "tiff" in ct:
            return ".tiff"
        if "webp" in ct:
            return ".webp"
    return ".bin"

def is_pdf_file(file_path: str):
    try:
        with open(file_path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False

def is_doc_file(file_path: str):
    ext = os.path.splitext(file_path.lower())[1]
    return ext in [".doc", ".docx"]

def cleanup_temp_path(path: str):
    try:
        if not os.path.exists(path):
            return
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        logger.info(f"Cleaned up temp path: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up path {path}: {e}")

def convert_office_to_pdf(input_path: str):
    if not shutil.which("soffice"):
        raise HTTPException(
            status_code=500,
            detail="DOC/DOCX conversion is not available because LibreOffice (soffice) is not installed"
        )
    output_dir = tempfile.mkdtemp(prefix="office_to_pdf_")
    try:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                output_dir,
                input_path
            ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        message = stderr or stdout or "Unknown LibreOffice conversion error"
        raise HTTPException(status_code=500, detail=f"DOC/DOCX conversion failed: {message}")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_pdf = os.path.join(output_dir, f"{base_name}.pdf")
    if not os.path.exists(output_pdf):
        raise HTTPException(status_code=500, detail="DOC/DOCX conversion failed: output PDF not found")
    return output_pdf, output_dir

def polygon_to_bbox(polygon):
    if not isinstance(polygon, list) or len(polygon) == 0:
        return None
    xs = []
    ys = []
    for point in polygon:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs or not ys:
        return None
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": x_max - x_min,
        "height": y_max - y_min
    }

ocr_engines = {"en": ocr}

def normalize_lang(lang: Optional[str]):
    if not lang:
        return "auto"
    value = lang.strip().lower()
    if value in ["zh", "cn", "chinese", "ch", "zh-cn", "zh_cn", "zh-hans"]:
        return "ch"
    if value in ["en", "english"]:
        return "en"
    if value in ["auto", "detect"]:
        return "auto"
    return "en"

def get_ocr_engine(lang: str):
    if lang in ocr_engines:
        return ocr_engines[lang]
    if lang == "ch":
        ocr_engines["ch"] = PaddleOCR(use_angle_cls=True, lang="ch", enable_mkldnn=False)
        logger.info("Chinese PaddleOCR initialized successfully.")
        return ocr_engines["ch"]
    return ocr

def process_page_positions_with_engine(
    page_num: int,
    image_path: str,
    ocr_engine,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
):
    result = None
    scale_x = 1.0
    scale_y = 1.0
    try:
        result = ocr_engine.ocr(image_path)
    except Exception as first_error:
        if not is_memory_error(first_error):
            raise
        image = decode_image_file(image_path)
        if image is None:
            raise
        original_height, original_width = image.shape[:2]
        retry_max_sides = [2000, 1600, 1280, 1024, 800, 640]
        last_error = first_error
        for max_side in retry_max_sides:
            resized = normalize_image_for_ocr(image, max_side=max_side)
            if resized is None:
                continue
            resized_height, resized_width = resized.shape[:2]
            if resized_width >= original_width and resized_height >= original_height:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as resized_img:
                cv2.imwrite(resized_img.name, resized)
                resized_path = resized_img.name
            try:
                result = ocr_engine.ocr(resized_path)
                scale_x = float(original_width) / float(resized_width)
                scale_y = float(original_height) / float(resized_height)
                break
            except Exception as retry_error:
                last_error = retry_error
                if not is_memory_error(retry_error):
                    cleanup_temp_file(resized_path)
                    raise
            finally:
                cleanup_temp_file(resized_path)
        if result is None:
            raise last_error
    page_data = process_ocr_result(result)
    try:
        page_data.sort(key=lambda x: (x["box"][0][1], x["box"][0][0]))
    except Exception:
        pass
    positions = []
    for item in page_data:
        polygon = scale_polygon(item.get("box", []), scale_x, scale_y)
        bbox = polygon_to_bbox(polygon)
        positions.append({
            "text": item.get("text", ""),
            "score": item.get("score", 0.0),
            "polygon": polygon,
            "bbox": bbox
        })
    return {
        "page": page_num,
        "width": image_width,
        "height": image_height,
        "results": positions
    }

def decode_image_file(image_path: str):
    with open(image_path, "rb") as f:
        raw = f.read()
    np_data = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return image

def normalize_image_for_ocr(image, max_side: int = 2500):
    if image is None:
        return None
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image
    scale = max_side / float(largest)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def prepare_image_png_for_ocr(source_path: str, max_side: int = 2500):
    image = decode_image_file(source_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    normalized = normalize_image_for_ocr(image, max_side=max_side)
    if normalized is None:
        raise HTTPException(status_code=400, detail="Failed to prepare image for OCR")
    height, width = normalized.shape[:2]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as normalized_img:
        ok = cv2.imwrite(normalized_img.name, normalized)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to convert image to PNG for OCR")
        return normalized_img.name, width, height

def scale_polygon(polygon, scale_x: float, scale_y: float):
    if not isinstance(polygon, list):
        return polygon
    scaled = []
    for point in polygon:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            scaled.append([float(point[0]) * scale_x, float(point[1]) * scale_y])
        else:
            scaled.append(point)
    return scaled

def is_memory_error(err: Exception):
    message = str(err)
    return "ResourceExhaustedError" in message or "Fail to alloc memory" in message

def process_page_positions_task(
    page_num: int,
    image_path: str,
    image_width: int,
    image_height: int,
    primary_engine,
    fallback_engine
):
    page_data = process_page_positions_with_engine(
        page_num,
        image_path,
        primary_engine,
        image_width,
        image_height
    )
    if fallback_engine and not page_data["results"]:
        page_data = process_page_positions_with_engine(
            page_num,
            image_path,
            fallback_engine,
            image_width,
            image_height
        )
    return page_data

def extract_positions_from_pdf(pdf_path: str, primary_engine, fallback_engine, temp_image_paths):
    page_tasks = []
    with fitz.open(pdf_path) as doc:
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                pix.save(temp_img.name)
                temp_image_paths.append(temp_img.name)
                page_tasks.append((page_idx + 1, temp_img.name, pix.width, pix.height))
    if not page_tasks:
        return []
    max_workers_raw = os.getenv("OCR_POSITIONS_MAX_WORKERS", "").strip()
    if max_workers_raw.isdigit() and int(max_workers_raw) > 0:
        max_workers = min(int(max_workers_raw), len(page_tasks))
    else:
        max_workers = len(page_tasks)
    pages = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(
                process_page_positions_task,
                page_num,
                image_path,
                image_width,
                image_height,
                primary_engine,
                fallback_engine
            ): page_num
            for page_num, image_path, image_width, image_height in page_tasks
        }
        for future in concurrent.futures.as_completed(future_to_page):
            page_data = future.result()
            pages.append(page_data)
    pages.sort(key=lambda item: item["page"])
    return pages

def cleanup_temp_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Cleaned up temp file: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {path}: {e}")

def organize_into_rows(ocr_data, y_threshold=10):
    """
    Groups OCR results into rows based on Y-coordinates.
    ocr_data: list of {'box': [[x1,y1]...], 'text': str, 'score': float}
    """
    if not ocr_data:
        return []

    # Sort by Top-Y
    sorted_data = sorted(ocr_data, key=lambda x: x['box'][0][1])
    
    rows = []
    current_row = []
    
    if sorted_data:
        current_row.append(sorted_data[0])
        
    for item in sorted_data[1:]:
        last_item = current_row[-1]
        
        # Calculate vertical center
        y1_curr = item['box'][0][1]
        y2_curr = item['box'][2][1]
        center_y_curr = (y1_curr + y2_curr) / 2
        
        y1_last = last_item['box'][0][1]
        y2_last = last_item['box'][2][1]
        center_y_last = (y1_last + y2_last) / 2
        
        height_last = y2_last - y1_last
        
        # Check if the item belongs to the current row (based on center Y overlap)
        if abs(center_y_curr - center_y_last) < (height_last * 0.5):
            current_row.append(item)
        else:
            # Sort current row by X coordinate (left to right)
            current_row.sort(key=lambda x: x['box'][0][0])
            rows.append(current_row)
            current_row = [item]
            
    if current_row:
        current_row.sort(key=lambda x: x['box'][0][0])
        rows.append(current_row)
        
    return rows

def detect_table_structure_cv2(image_path):
    """
    Detects table structure (rows and columns) using OpenCV grid line detection.
    Returns a list of cell bounding boxes sorted by position.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines to get grid
        mask = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0.0)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Find contours (potential cells)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter out very small contours or very large (whole page)
            if w > 20 and h > 10 and w < img.shape[1] * 0.9 and h < img.shape[0] * 0.9:
                cells.append([x, y, w, h])
        
        # Sort cells top-to-bottom, left-to-right
        # Sorting by Y first with a tolerance for row alignment
        cells.sort(key=lambda b: (b[1] // 10, b[0]))
        
        return cells
    except Exception as e:
        logger.error(f"Error in CV2 table detection: {e}")
        return []

def extract_text_from_cells(image_path, cells):
    """
    Extracts text from each detected cell using PaddleOCR.
    """
    img = cv2.imread(image_path)
    table_data = []
    
    # Sort cells into rows
    rows = []
    if not cells:
        return []
        
    # Simple row clustering
    current_row = [cells[0]]
    for i in range(1, len(cells)):
        prev_y = current_row[-1][1]
        curr_y = cells[i][1]
        prev_h = current_row[-1][3]
        
        if abs(curr_y - prev_y) < (prev_h * 0.5):
            current_row.append(cells[i])
        else:
            current_row.sort(key=lambda b: b[0])
            rows.append(current_row)
            current_row = [cells[i]]
    if current_row:
        current_row.sort(key=lambda b: b[0])
        rows.append(current_row)
    
    # Extract text from each cell
    for row_idx, row_cells in enumerate(rows):
        row_data = []
        for col_idx, (x, y, w, h) in enumerate(row_cells):
            # Crop cell
            roi = img[y:y+h, x:x+w]
            
            # OCR on cell
            # Using basic OCR here since we cropped to cell
            try:
                # Add padding to ROI to improve OCR accuracy
                # PaddleOCR sometimes fails on tight crops
                h_img, w_img, _ = img.shape
                pad = 5
                y1 = max(0, y - pad)
                y2 = min(h_img, y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(w_img, x + w + pad)
                roi = img[y1:y2, x1:x2]

                # The 'cls' argument was causing errors on some PaddleOCR versions
                # when called on a cropped image (numpy array).
                # We can try calling it without cls=False first.
                result = ocr.ocr(roi)
                cell_text = ""
                if result and result[0]:
                    # Filter out low confidence and very small text
                    # Join valid text parts
                    texts = []
                    for line in result[0]:
                         if line:
                             text_content = line[1][0]
                             # Fix common OCR issue where it returns single chars separated by spaces
                             # e.g. "n a o t o" -> "naoto" if it looks like garbage
                             if len(text_content) > 3 and text_content.count(' ') > len(text_content) / 2:
                                 text_content = text_content.replace(' ', '')
                             texts.append(text_content)
                    cell_text = " ".join(texts)
                
                row_data.append({
                    "row": row_idx,
                    "col": col_idx,
                    "text": cell_text.strip(),
                    "bbox": [x, y, w, h]
                })
            except Exception as e:
                logger.warning(f"OCR failed for cell at {x},{y}: {e}")
                row_data.append({"text": "", "bbox": [x, y, w, h], "error": str(e)})
        
        if row_data:
            table_data.append(row_data)
            
    return table_data

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

    # Handle list of dicts (new format, common in newer PaddleOCR versions)
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        for res in result:
            # Check if it has parallel lists (rec_texts, rec_scores, dt_polys)
            if 'rec_texts' in res and isinstance(res['rec_texts'], list):
                texts = res['rec_texts']
                # Prefer 'dt_polys' (detection polygons) or 'rec_polys' (recognition polygons)
                # dt_polys seems to be the standard key for boxes in this format
                boxes = res.get('dt_polys', [])
                if not boxes:
                    boxes = res.get('rec_polys', [])
                
                scores = res.get('rec_scores', [])
                
                # Zip them together if lengths match (or truncating to shortest)
                limit = min(len(texts), len(boxes))
                for i in range(limit):
                    item = {
                        "box": convert_numpy_types(boxes[i]),
                        "text": texts[i],
                        "score": convert_numpy_types(scores[i]) if i < len(scores) else 0.0
                    }
                    processed_result.append(item)
            else:
                # Fallback: just append the dict if it doesn't match known structure
                # But filter large fields
                processed_result.append(convert_numpy_types(res))
            
    # Handle legacy format (list of lists: [[[box], [text, score]], ...])
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
    url = sanitize_input_url(request.url)
    logger.info(f"Received request for URL: {url}")

    temp_pdf_path = None
    
    try:
        # Download PDF
        response = requests.get(
            url,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 OCR-Service/1.0", "Accept": "*/*"}
        )
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

        # Process OCR result
        page_data = process_ocr_result(result)
        
        # Sort by Y coordinate first (top to bottom), then X (left to right)
        # Bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        # We sort by y1 of the first point
        try:
            page_data.sort(key=lambda x: (x['box'][0][1], x['box'][0][0]))
        except Exception:
            pass # Keep original order if sorting fails

        # Organize into table-like rows
        try:
            structured_rows = organize_into_rows(page_data)
        except Exception as e:
            logger.warning(f"Failed to organize rows for page {page_num}: {e}")
            structured_rows = []

        return {
            "page": page_num,
            "result": structured_rows,  # Return structured rows as the main result
            "raw_data": page_data       # Keep flat list if needed
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
    # Try img2table with PaddleOCR first
    if IMG2TABLE_AVAILABLE:
        try:
            logger.info(f"Page {page_num}: Attempting img2table extraction...")
            # Initialize img2table's PaddleOCR wrapper
            # We use 'en' language by default
            # Explicitly set enable_mkldnn=False to avoid crash
            # Removed use_angle_cls=True as it caused TypeError in img2table
            ocr_img2table = Img2TablePaddleOCR(lang='en', enable_mkldnn=False)
            
            # Create Image object
            doc = Img2TableImage(src=temp_img_path)
            
            # Extract tables
            # implicit_rows=False helps avoid merging separate text lines into one row incorrectly
            extracted_tables = doc.extract_tables(ocr=ocr_img2table, implicit_rows=False, borderless_tables=False, min_confidence=50)
            
            if extracted_tables:
                logger.info(f"Page {page_num}: img2table detected {len(extracted_tables)} tables.")
                
                # Format result
                tables_data = []
                for table_idx, table in enumerate(extracted_tables):
                    table_content = []
                    # table.df is a pandas dataframe, but we might want raw cells for bbox info
                    # table.content is a dict of rows
                    
                    # Convert to our format: list of rows, each row is list of cells
                    # We need to handle potential sparse rows
                    
                    # Get all rows sorted
                    rows = sorted(table.content.keys())
                    for row_id in rows:
                        row_cells = table.content[row_id]
                        formatted_row = []
                        # Sort cells in row by column index
                        cols = sorted(row_cells)
                        for cell in cols:
                            cell_obj = row_cells[cell]
                            formatted_row.append({
                                "text": cell_obj.value,
                                "bbox": [cell_obj.bbox.x1, cell_obj.bbox.y1, cell_obj.bbox.x2 - cell_obj.bbox.x1, cell_obj.bbox.y2 - cell_obj.bbox.y1],
                                "row": row_id,
                                "col": cell_obj.col
                            })
                        table_content.append(formatted_row)
                    
                    tables_data.append({
                        "table_index": table_idx,
                        "content": table_content,
                        "html": table.html
                    })
                
                return {
                    "page": page_num,
                    "result": tables_data,
                    "method": "img2table"
                }
            else:
                 logger.info(f"Page {page_num}: img2table found no tables.")
        except Exception as e:
            logger.error(f"Page {page_num}: img2table extraction failed: {e}")

    # Try OpenCV Table Detection first (Grid-based)
    try:
        cells = detect_table_structure_cv2(temp_img_path)
        if cells and len(cells) > 5: # Assuming a valid table has at least a few cells
            logger.info(f"Page {page_num}: Detected {len(cells)} table cells using OpenCV. Extracting text...")
            table_data = extract_text_from_cells(temp_img_path, cells)
            return {
                "page": page_num,
                "result": table_data,
                "method": "opencv_grid_detection"
            }
        else:
            logger.info(f"Page {page_num}: No significant grid detected. Falling back to standard OCR.")
    except Exception as e:
        logger.error(f"Page {page_num}: OpenCV table detection failed: {e}")

    # Fallback to standard OCR if PPStructure is not available
    if not table_engine:
        logger.warning(f"PPStructure not initialized, falling back to standard OCR for page {page_num}")
        return process_page_ocr(page_num, temp_img_path)

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
    url = sanitize_input_url(request.url)
    logger.info(f"Received request for URL: {url} (extract-table-text)")

    temp_pdf_path = None
    
    try:
        # Download PDF
        response = requests.get(
            url,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 OCR-Service/1.0", "Accept": "*/*"}
        )
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

@app.post("/extract-text-positions")
async def extract_text_positions(request: PDFRequest, background_tasks: BackgroundTasks):
    url = sanitize_input_url(request.url)
    lang = normalize_lang(request.lang)
    logger.info(f"Received request for URL: {url} (extract-text-positions, lang={lang})")
    temp_file_path = None
    temp_converted_pdf_path = None
    temp_conversion_dir = None
    temp_image_paths = []
    try:
        response = requests.get(
            url,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 OCR-Service/1.0", "Accept": "*/*"}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download file from URL")
        suffix = guess_file_suffix(url, response.headers.get("content-type"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        primary_engine = get_ocr_engine("en" if lang == "auto" else lang)
        fallback_engine = get_ocr_engine("ch") if lang == "auto" else None
        if is_doc_file(temp_file_path):
            temp_converted_pdf_path, temp_conversion_dir = convert_office_to_pdf(temp_file_path)
            pages = extract_positions_from_pdf(temp_converted_pdf_path, primary_engine, fallback_engine, temp_image_paths)
            return {"status": "success", "file_type": "doc", "data": pages}
        if is_pdf_file(temp_file_path):
            pages = extract_positions_from_pdf(temp_file_path, primary_engine, fallback_engine, temp_image_paths)
            return {"status": "success", "file_type": "pdf", "data": pages}
        normalized_png_path, width, height = prepare_image_png_for_ocr(temp_file_path, max_side=2500)
        temp_image_paths.append(normalized_png_path)
        page_data = process_page_positions_with_engine(1, normalized_png_path, primary_engine, width, height)
        if lang == "auto" and not page_data["results"]:
            fallback_engine = get_ocr_engine("ch")
            page_data = process_page_positions_with_engine(1, normalized_png_path, fallback_engine, width, height)
        return {"status": "success", "file_type": "image", "data": [page_data]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request in extract-text-positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        if temp_converted_pdf_path:
            background_tasks.add_task(cleanup_temp_file, temp_converted_pdf_path)
        if temp_conversion_dir:
            background_tasks.add_task(cleanup_temp_path, temp_conversion_dir)
        for temp_img_path in temp_image_paths:
            background_tasks.add_task(cleanup_temp_file, temp_img_path)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
