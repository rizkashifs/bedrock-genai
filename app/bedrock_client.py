import boto3
import os
from dotenv import load_dotenv
import json

# Load environment variables first
load_dotenv()

# Use the correct model ID for Claude 3.5 Sonnet
#modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
modelId = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
print(modelId)

# Debug: Check if credentials are loaded
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

print(f"AWS Access Key loaded: {'Yes' if aws_access_key else 'No'}")
print(f"AWS Secret Key loaded: {'Yes' if aws_secret_key else 'No'}")
print(f"AWS Region: {aws_region}")


# Initialize bedrock client - get credentials from environment variables
bedrock = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')  # Default to us-east-1 if not set
)

def invoke_claude(messages: list) -> str:
    import boto3
import os
from dotenv import load_dotenv
import json

# Load environment variables first
load_dotenv()

# Use the correct model ID for Claude
modelId = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
print("Using model:", modelId)

# Debug: Check if credentials are loaded
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

#print(f"AWS Access Key loaded: {'Yes' if aws_access_key else 'No'}")
#print(f"AWS Secret Key loaded: {'Yes' if aws_secret_key else 'No'}")
#print(f"AWS Region: {aws_region}")

# Initialize bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

def invoke_claude(prompt: str) -> str:
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
                "maxTokens": 200, 
                "temperature": 0,  # More natural responses
                "topP": 0
            }
        )
        
        # Extract the response text
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text
        
    except Exception as e:
        return f"Error invoking Claude: {str(e)}"

def invoke_claude_messages(messages: list) -> str:
    """
    Invoke Claude via AWS Bedrock with a list of messages.
    Each message should be a dict with 'role' and 'content'.
    Example:
    [{"role": "user", "content": [{"text": "Hello"}]}]
    """
    # Structure the conversation properly as a list
    #conversation = [
    #    {
    #        "role": "user", 
    #        "content": [{"text": prompt}]
    #    }
    #]
    
    try:
        response = bedrock.converse(
            modelId=modelId,
            messages=messages,
            inferenceConfig={
                "maxTokens": 200, 
                "temperature": 0,  # More natural responses
                "topP": 0
            }
        )
        
        # Extract the response text
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text
        
    except Exception as e:
        return f"Error invoking Claude: {str(e)}"

# Example usage
if __name__ == "__main__":
   test_messages = [
        {
            "role": "user",
            "content": [{"text": "Tell me a joke about cloud computing."}]
        }
    ]
   result = invoke_claude(test_messages)
   print("Claude says:", result)