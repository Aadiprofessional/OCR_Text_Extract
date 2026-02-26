import os
import sys

# Set cache dir to a writable location
os.environ['PADDLEOCR_CACHE_DIR'] = '/tmp/paddleocr_cache'
os.environ['PADDLEX_HOME'] = '/tmp/paddlex'
os.environ['PADDLE_PDX_CACHE_HOME'] = '/tmp/paddlex_home'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

try:
    from paddleocr import PaddleOCR, PPStructure
except ImportError:
    try:
        from paddleocr import PaddleOCR, PPStructureV2 as PPStructure
    except ImportError:
         # Check if we should use PPStructureV3
         try:
             from paddleocr import PPStructureV3 as PPStructure
         except ImportError:
              print("PPStructureV3 also not found.")
              # Fallback to None or raise error later
              PPStructure = None

if 'PPStructure' not in locals() or PPStructure is None:
    # Try importing from submodules if needed
    pass

try:
    print("Imported successfully")
    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    print("OCR Initialized")
    
    # Initialize Table Structure
    if PPStructure:
        print(f"Initializing {PPStructure.__name__}...")
        try:
             # Try standard init
             table_engine = PPStructure(show_log=True, image_orientation=True)
        except TypeError:
             # PPStructureV3 might have different args
             print("Standard init failed, trying PPStructureV3 args...")
             table_engine = PPStructure(use_doc_orientation_classify=True, use_doc_unwarping=False)
        
        print("PPStructure Initialized")
    else:
        print("PPStructure class not found, skipping table structure model download.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
