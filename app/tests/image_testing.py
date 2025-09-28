import boto3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import base64

# Assuming 'util.py' exists and provides 'logger' and 'modelId'

from app.utils.util import logger, modelId
from app.models.bedrock_client import invoke_claude_messages # Import the function you provided

# --- Configuration ---
IMAGE_PATH = r"C:\Users\admin\Downloads\bedrock-testing\app\test.png" #  UPDATE THIS PATH 
USER_QUESTION = "Describe the contents of this image and identify any text you see."
# ---------------------

def encode_image_to_bytes(file_path: str) -> bytes:
    """Read image file and return raw bytes."""
    try:
        with open(file_path, 'rb') as image_file:
            image_bytes = image_file.read()

        if not image_bytes:
            logger.error(f"Image file {file_path} was empty.")
            sys.exit(1)
            
        logger.info(f"Image loaded as bytes: {file_path}")
        return image_bytes
 
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        sys.exit(1)


def create_multimodal_message(image_bytes: bytes, file_path: str, question: str) -> list:
    """
    Creates the correct Bedrock Converse API message structure for multimodal input.
    The structure is: [
        {
            "role": "user",
            "content": [
                {"image": {"format": "png", "source": {"bytes": <bytes>}}},
                {"text": "Your question"}
            ]
        }
    ]
    """
    file_extension = Path(file_path).suffix.lower().lstrip('.')
    
    # Ensure a supported media type is used
    if file_extension not in ['png', 'jpeg', 'jpg']:
        media_type = 'jpeg' # Default fallback, though should be checked
    else:
        media_type = 'png' if file_extension == 'png' else 'jpeg'

    # The multimodal content parts list
    content_parts = [
        # 1. Image part (must contain raw bytes)
        { 
            "image" : { 
                "format": media_type,
                "source": {
                    "bytes": image_bytes 
                }
            }
        },
        # 2. Text part
        {
            "text": f"Here is the file '{Path(file_path).name}'. {question}"
        }
    ]

    # The final message list for the API is a single user message
    # containing all the content parts.
    messages_payload = [
        {
            "role": "user",
            "content": content_parts
        }
    ]
    
    return messages_payload


def test_image_invocation():
    """Main function to run the image invocation test."""
    logger.info("Starting Multimodal Image Invocation Test...")

    # 1. Load and encode the image
    if not os.path.exists(IMAGE_PATH):
        logger.error(f"Image file not found at: {IMAGE_PATH}")
        sys.exit(1)

    image_bytes = encode_image_to_bytes(IMAGE_PATH)

    # 2. Create the correctly structured messages payload
    messages_to_send = create_multimodal_message(
        image_bytes=image_bytes,
        file_path=IMAGE_PATH,
        question=USER_QUESTION
    )

    # Log the structure (without crashing on bytes)
    try:
        # Create a serializable version for logging
        debug_payload = json.loads(json.dumps(messages_to_send))
        # Replace the large bytes object with a placeholder string
        debug_payload[0]["content"][0]["image"]["source"]["bytes"] = f"<RAW_BYTES: size={len(image_bytes)}>"
        logger.info("\n=== PAYLOAD STRUCTURE ===")
        logger.info(json.dumps(debug_payload, indent=2))
        logger.info("=========================\n")
    except Exception as e:
        logger.warning(f"Failed to print debug payload: {e}")


    # 3. Invoke Claude with the multimodal messages
    logger.info(f"Invoking Claude with question: {USER_QUESTION}")
    
    # Use your existing invoke_claude_messages function
    response = invoke_claude_messages(
        messages=messages_to_send, 
        temperature=0.1, 
        max_tokens=500
    )

    # 4. Print the final result
    logger.info("--- CLAUDE RESPONSE ---")
    print(response)
    logger.info("-----------------------")


if __name__ == "__main__":
    test_image_invocation()