import requests
import json
import time

def test_api():
    url = "https://python.matrixedu.ai/extract-text"
    payload = {
        "url": "https://www.princexml.com/samples/catalogue/PrinceCatalogue.pdf"
    }
    
    print(f"Testing API at {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Time Taken: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print("\nResponse:")
            # Print first 500 chars of result to avoid spamming console
            print(json.dumps(data, indent=2)[:1000] + "...")
        else:
            print("\nError Response:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API. Is it running on localhost:8000?")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    test_api()
