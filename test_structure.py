import os
import cv2
from paddleocr import PPStructure, save_structure_res

def test_structure():
    try:
        # Initialize PPStructure
        # table=True enables table recognition
        # ocr=True enables OCR in table cells
        table_engine = PPStructure(show_log=True, image_orientation=True)
        print("PPStructure initialized successfully.")
        
        # We need a sample image. 
        # Since I don't have the user's PDF downloaded locally in a persistent way,
        # I will just check if the class initializes. 
        # If I really need to test, I can download a small image.
        
    except Exception as e:
        print(f"Failed to initialize PPStructure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_structure()
