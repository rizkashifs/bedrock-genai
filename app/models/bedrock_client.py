import boto3
import os
from dotenv import load_dotenv
import json
from app.utils.util import logger, modelId

# Load environment variables first
load_dotenv()

# Print Model being used
logger.info("Using model: %s", modelId)

# Debug: Check if credentials are loaded
'''
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

print(f"AWS Access Key loaded: {'Yes' if aws_access_key else 'No'}")
print(f"AWS Secret Key loaded: {'Yes' if aws_secret_key else 'No'}")
print(f"AWS Region: {aws_region}")
'''

# Initialize bedrock client - get credentials from environment variables
bedrock = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-2')  # Default to us-east-2 if not set
)

def invoke_claude(prompt: str, temperature: float = 0, max_tokens: int = 200,top_p: float = 0) -> str:
    """
    Invoke Claude via AWS Bedrock with a list of messages.
    Each message should be a dict with 'role' and 'content'.
    Example:
    [{"role": "user", "content": [{"text": "Hello"}]}]
    """
    # Structure the conversation properly as a list
    conversation = [
        {
            "role": "user", 
            "content": [{"text": prompt}]
        }
    ]
    
    try:
        response = bedrock.converse(
            modelId=modelId,
            messages=conversation,
            inferenceConfig={
                "maxTokens": max_tokens, 
                "temperature": temperature,  # More natural responses
                "topP": top_p
            }
        )
        
        # Extract the response text
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text
        
    except Exception as e:
        logger.error("Claude invocation failed: %s", str(e))
        return f"Error invoking Claude: {str(e)}"

def invoke_claude_messages(messages: list, temperature: float = 0, max_tokens: int = 200,top_p: float = 0) -> str:
    """
    Invoke Claude via AWS Bedrock with a list of messages.
    Each message should be a dict with 'role' and 'content'.
    Example:
    [{"role": "user", "content": [{"text": "Hello"}]}]
    """    
    try:
        response = bedrock.converse(
            modelId=modelId,
            messages=messages,
            inferenceConfig={
                "maxTokens": max_tokens, 
                "temperature": temperature,  # More natural responses
                "topP": top_p
            }
        )
        
        # Extract the response text
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text
        
    except Exception as e:
        logger.error("Claude invocation failed: %s", str(e))
        return f"Error invoking Claude: {str(e)}"

# Example usage
if __name__ == "__main__":
   test_messages = [
        {
            "role": "user",
            "content": [{"text": "Tell me a joke about cloud computing."}]
        }
    ]
   result = invoke_claude_messages(test_messages)
   print("Claude says:", result)