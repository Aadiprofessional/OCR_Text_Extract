import requests
import json
import time

def test_api_position():
    # Change to your server URL if different
    url = "http://localhost:8000/extract-text-with-position"
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
            print("\nResponse Data Structure (First Page Result):")
            if data.get("data") and len(data["data"]) > 0:
                first_page = data["data"][0]
                print(f"Page: {first_page.get('page')}")
                print("First 2 items in result:")
                print(json.dumps(first_page.get("result", [])[:2], indent=2))
            else:
                print("No data returned.")
        else:
            print("\nError Response:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API. Is it running on localhost:8000?")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    test_api_position()
