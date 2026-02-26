import os
import tempfile
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PPStructureV3
from pdf2image import convert_from_path
from html.parser import HTMLParser

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading PPStructureV3 model...")
    pipeline = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print("Model loaded successfully!")
    yield

app = FastAPI(title="PP-StructureV3 OCR API", lifespan=lifespan)

class ExtractRequest(BaseModel):
    url: str
    pages: list[int] | None = None  # e.g. [0, 1] — None means all pages

@app.get("/health")
def health():
    return {"status": "ok", "model": "PPStructureV3", "loaded": pipeline is not None}

@app.post("/extract")
def extract(req: ExtractRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # 1. Download PDF
    try:
        resp = requests.get(req.url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

    # 2. Save PDF to temp
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        # 3. Convert PDF pages to images
        all_images = convert_from_path(tmp_path, dpi=150)

        # 4. Select pages
        if req.pages is not None:
            selected = [(i, all_images[i]) for i in req.pages if i < len(all_images)]
        else:
            selected = list(enumerate(all_images))

        results = []

        for page_index, image in selected:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_tmp:
                image.save(img_tmp.name)
                img_path = img_tmp.name

            try:
                ocr_output = pipeline.predict(img_path)

                page_result = {
                    "page": page_index,
                    "tables": [],
                    "text_blocks": []
                }

                for res in ocr_output:
                    res_dict = res.json if hasattr(res, "json") else {}

                    # Extract tables
                    for table in res_dict.get("table_res_list", []):
                        html = table.get("pred_html", "")
                        cells = parse_html_to_cells(html)
                        page_result["tables"].append({
                            "bbox": table.get("bbox", []),
                            "html": html,
                            "cells": cells
                        })

                    # Extract text blocks
                    layout_boxes = res_dict.get("layout_det_res", {}).get("boxes", [])
                    for box in layout_boxes:
                        label = box.get("label", "")
                        if label in ["text", "paragraph_title", "doc_title", "figure_title"]:
                            page_result["text_blocks"].append({
                                "type": label,
                                "bbox": box.get("coordinate", []),
                                "text": box.get("text", "")
                            })

                results.append(page_result)

            finally:
                os.unlink(img_path)

    finally:
        os.unlink(tmp_path)

    return {
        "status": "success",
        "total_pages": len(results),
        "data": results
    }


def parse_html_to_cells(html: str):
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
                self.cells.append({
                    "row": self.current_row,
                    "col": self.current_col,
                    "text": self.current_text.strip()
                })
                self.current_col += 1
                self.in_cell = False

        def handle_data(self, data):
            if self.in_cell:
                self.current_text += data

    p = TableParser()
    p.feed(html)
    return p.cells