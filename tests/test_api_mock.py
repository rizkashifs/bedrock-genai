import sys
from unittest.mock import MagicMock, patch

# Mock boto3 before importing app to avoid AWS connection attempts
mock_boto3 = MagicMock()
sys.modules["boto3"] = mock_boto3
mock_bedrock = MagicMock()
mock_boto3.client.return_value = mock_bedrock

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint():
    print("Testing /chat endpoint with mocks...")
    
    # Mock the ChatEngine to avoid AWS calls and DynamoDB
    with patch('app.services.chat_orchestration.ChatEngine') as MockChatEngine:
        # Setup mock instance
        mock_instance = MockChatEngine.return_value
        mock_instance.send_message.return_value = "This is a mocked response from Claude."
        mock_instance.get_complete_history.return_value = [
            {"user": "Hello", "assistant": "Hi"},
            {"user": "Test", "assistant": "Response"}
        ]
        
        payload = {
            "message": "Test message",
            "system_prompt": "You are a test bot.",
            "feature": "qna"
        }
        
        response = client.post("/chat", json=payload)
        
        if response.status_code == 200:
            print("Success! Status 200 OK")
            data = response.json()
            print(f"Response: {data['response']}")
            print(f"History items: {len(data['history'])}")
            assert data['response'] == "This is a mocked response from Claude."
            assert len(data['history']) == 2
        else:
            print(f"Failed: {response.status_code}")
            print(response.text)
            sys.exit(1)

if __name__ == "__main__":
    test_chat_endpoint()
