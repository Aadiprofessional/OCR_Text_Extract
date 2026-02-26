import io 
import json 
import requests 
import tempfile 
import os 
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
from paddleocr import PPStructureV3 
from pdf2image import convert_from_path 
from PIL import Image 

app = FastAPI(title="PP-StructureV3 OCR API") 

# Load pipeline once at startup (CPU mode — no GPU needed) 
pipeline = PPStructureV3() 

class ExtractRequest(BaseModel): 
    url: str 
    pages: list[int] | None = None  # e.g. [0, 1, 2] — None means all pages 

@app.get("/health") 
def health(): 
    return {"status": "ok"} 

@app.post("/extract") 
def extract(req: ExtractRequest): 
    # 1. Download the PDF from URL 
    try: 
        response = requests.get(req.url, timeout=30) 
        response.raise_for_status() 
    except Exception as e: 
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}") 

    # 2. Save PDF to temp file 
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp: 
        tmp.write(response.content) 
        tmp_path = tmp.name 

    try: 
        # 3. Convert PDF pages to images 
        images = convert_from_path(tmp_path, dpi=150) 

        # 4. Filter pages if specified 
        if req.pages: 
            images = [images[i] for i in req.pages if i < len(images)] 

        results = [] 

        # 5. Run PP-StructureV3 on each page 
        for page_num, image in enumerate(images): 
            page_result = { 
                "page": req.pages[page_num] if req.pages else page_num, 
                "tables": [], 
                "text_blocks": [] 
            } 

            # Save PIL image to temp file for PaddleOCR 
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_tmp: 
                image.save(img_tmp.name) 
                img_path = img_tmp.name 

            # Run OCR 
            ocr_output = pipeline.predict(img_path) 

            for res in ocr_output: 
                res_json = res.json  # get structured JSON output 

                for item in res_json.get("layout_det_res", {}).get("boxes", []): 
                    label = item.get("label", "") 
                    bbox = item.get("coordinate", []) 

                    if label == "table": 
                        # Find matching table content 
                        table_data = extract_table_from_result(res_json, bbox) 
                        page_result["tables"].append({ 
                            "bbox": bbox, 
                            "html": table_data.get("html", ""), 
                            "cells": table_data.get("cells", []) 
                        }) 
                    elif label in ["text", "paragraph_title", "doc_title"]: 
                        page_result["text_blocks"].append({ 
                            "label": label, 
                            "bbox": bbox, 
                            "text": item.get("text", "") 
                        }) 

            os.unlink(img_path) 
            results.append(page_result) 

    finally: 
        os.unlink(tmp_path) 

    return {"status": "success", "total_pages": len(results), "data": results} 


def extract_table_from_result(res_json, table_bbox): 
    """Extract table cells and HTML from PP-StructureV3 result""" 
    cells = [] 
    html = "" 

    table_res = res_json.get("table_res_list", []) 
    for table in table_res: 
        html = table.get("pred_html", "") 
        cell_bbox_list = table.get("cell_bbox", []) 
        # Convert HTML table to structured cells 
        from html.parser import HTMLParser 
        cells = parse_html_table_to_cells(html, cell_bbox_list) 
        break 

    return {"html": html, "cells": cells} 


def parse_html_table_to_cells(html: str, bboxes: list): 
    """Parse HTML table into row/col cell structure with bboxes""" 
    from html.parser import HTMLParser 

    class TableParser(HTMLParser): 
        def __init__(self): 
            super().__init__() 
            self.cells = [] 
            self.current_row = -1 
            self.current_col = 0 
            self.in_cell = False 
            self.current_text = "" 

        def handle_starttag(self, tag, attrs): 
            if tag == "tr": 
                self.current_row += 1 
                self.current_col = 0 
            elif tag in ("td", "th"): 
                self.in_cell = True 
                self.current_text = "" 

        def handle_endtag(self, tag): 
            if tag in ("td", "th") and self.in_cell: 
                bbox_index = len(self.cells) 
                bbox = bboxes[bbox_index] if bbox_index < len(bboxes) else [] 
                self.cells.append({ 
                    "row": self.current_row, 
                    "col": self.current_col, 
                    "text": self.current_text.strip(), 
                    "bbox": bbox 
                }) 
                self.current_col += 1 
                self.in_cell = False 

        def handle_data(self, data): 
            if self.in_cell: 
                self.current_text += data 

    parser = TableParser() 
    parser.feed(html) 
    return parser.cells