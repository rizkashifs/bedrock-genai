import requests
import json
import sys

def test_chat_api():
    url = "http://localhost:8000/chat"
    
    payload = {
        "message": "Hello, who are you?",
        "system_prompt": "You are a helpful assistant.",
        "feature": "qna"
    }
    
    print(f"Testing API at {url}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Note: This requires the server to be running. 
        # In a CI/CD or automated env, we'd use TestClient from starlette/fastapi
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("\nSuccess!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"\nFailed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to server. Make sure it's running on localhost:8000")

if __name__ == "__main__":
    test_chat_api()
