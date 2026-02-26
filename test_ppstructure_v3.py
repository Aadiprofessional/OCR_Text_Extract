import os
import sys

# Set environment variable to bypass model source check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

try:
    from paddleocr import PPStructureV3
    print("PPStructureV3 imported successfully.")
except ImportError:
    print("Error: PPStructureV3 not found in paddleocr module.")
    print("Please ensure you have paddleocr>=2.8.0 installed.")
    sys.exit(1)

def test_ppstructure():
    print("Initializing PPStructureV3...")
    try:
        pipeline = PPStructureV3(device='cpu')
        print("PPStructureV3 initialized.")
        
        # You can add a test with a local file here if available
        # output = pipeline.predict("path/to/test.png")
        # print("Prediction successful.")
        
    except Exception as e:
        print(f"Failed to initialize PPStructureV3: {e}")

if __name__ == "__main__":
    test_ppstructure()
