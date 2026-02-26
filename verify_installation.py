import os
import sys
import traceback

def verify_installation():
    print("="*50)
    print("Verifying PP-StructureV3 Installation...")
    print("="*50)

    # Set environment variables for cache to avoid PermissionError
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_CACHE_HOME"] = os.path.join(os.getcwd(), ".paddlex_cache")
    os.environ["PADDLE_HOME"] = os.path.join(os.getcwd(), ".paddle_home")
    
    print(f"Cache directory set to: {os.environ['PADDLE_PDX_CACHE_HOME']}")
    
    try:
        print("Importing PPStructureV3...")
        from paddleocr import PPStructureV3
        print("Import successful.")
        
        print("Initializing PPStructureV3 (device='cpu')...")
        pipeline = PPStructureV3(device='cpu')
        print("Initialization successful!")
        print("="*50)
        print("PP-StructureV3 is ready to use.")
        return True
        
    except ImportError as e:
        print("\nERROR: Import failed.")
        print(f"Details: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print("\nERROR: Initialization failed.")
        print(f"Details: {e}")
        traceback.print_exc()
        
        if "dependency error" in str(e).lower():
            print("\nSUGGESTION: Try installing paddlex with OCR support:")
            print('pip install "paddlex[ocr]"')
            
        return False

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
