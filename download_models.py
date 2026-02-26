import logging
import traceback
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_downloader")

def download_models():
    logger.info("Starting model download check...")
    
    # 1. Try to download PaddleOCR models
    try:
        from paddleocr import PaddleOCR
        logger.info("Initializing PaddleOCR to trigger model download...")
        PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
        logger.info("PaddleOCR models checked/downloaded.")
    except Exception as e:
        logger.warning(f"Failed to initialize PaddleOCR: {e}")
        traceback.print_exc()

    # 2. Try to download PPStructure models
    try:
        # Import logic matching main.py
        try:
            from paddleocr import PPStructure
        except ImportError:
            try:
                from paddleocr.ppstructure import PPStructure
            except ImportError:
                try:
                    from ppstructure.predict_system import PPStructure
                except ImportError:
                    logger.warning("Could not import PPStructure from any known location.")
                    PPStructure = None

        if PPStructure:
            logger.info("Initializing PPStructure to trigger model download...")
            # image_orientation=False as we removed paddleclas
            PPStructure(show_log=False, image_orientation=False)
            logger.info("PPStructure models checked/downloaded.")
        else:
            logger.warning("Skipping PPStructure model download because import failed.")
            
    except Exception as e:
        logger.warning(f"Failed to initialize PPStructure: {e}")
        traceback.print_exc()

    logger.info("Model download process completed (success or partial failure ignored for build).")

if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        # Catch generic top-level errors to ensure exit code 0
        logger.error(f"Unexpected error in download script: {e}")
        traceback.print_exc()
    
    # Always exit 0 to allow Docker build to continue
    sys.exit(0)
