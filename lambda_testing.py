import json
from app.chat import ChatEngine, get_system_prompt

def lambda_handler(event, context):
    """
    Lambda handler for chat functionality.
    Always uses API mode for structured JSON responses.
    
    Expected event format:
    {
        "chat_id": "user_123_session_456",
        "message": "What is Python?",
        "feature": "qna",  # Optional: docComparison, qna, search, codeReview
        "user_id": "user_123",  # Optional metadata
        "auto_save": true  # Optional: whether to save to DynamoDB immediately
    }
    """
    try:
        # Extract required parameters
        chat_id = event['chat_id']
        user_message = event['message']
        
        # Extract optional parameters
        feature = event.get('feature')
        user_id = event.get('user_id')
        auto_save = event.get('auto_save', True)
        
        # Prepare metadata
        metadata = {}
        if user_id:
            metadata['user_id'] = user_id
        if feature:
            metadata['feature'] = feature
        
        # Get system prompt based on feature
        system_prompt = None
        if feature:
            system_prompt = get_system_prompt(feature, mode="api")  # Always API mode
        
        # Initialize ChatEngine in API mode (always structured responses)
        chat = ChatEngine(
            chat_id=chat_id,
            system_prompt=system_prompt,
            mode="api",  # Always API mode for Lambda
            use_dynamo=True,
            metadata=metadata
        )
        
        # Send message and get response
        response = chat.send_message(user_message)
        
        # Save to DynamoDB if requested
        if auto_save:
            save_success = chat.save_to_dynamo()
            if not save_success:
                print(f"Warning: Failed to save chat {chat_id} to DynamoDB")
        
        # Get conversation stats
        complete_history = chat.get_complete_history()
        session_history = chat.get_session_history()
        dynamo_history = chat.get_dynamo_history()
        
        # Return successful response with metadata
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # For CORS if needed
            },
            'body': json.dumps({
                'response': response,  # Claude's response (should be JSON if using API mode)
                'chat_id': chat_id,
                'message_count': len(complete_history),
                'session_exchanges': len(session_history),
                'saved_exchanges': len(dynamo_history),
                'auto_saved': auto_save,
                'feature': feature,
                'mode': 'api'
            })
        }
        
    except KeyError as e:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': f'Missing required parameter: {str(e)}',
                'required_fields': ['chat_id', 'message']
            })
        }
    
    except Exception as e:
        print(f"Lambda error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        }


def test_lambda_locally():
    """
    Test the lambda function locally with sample events.
    """
    print("Testing Lambda Function Locally")

    test_case = {
                "chat_id": "test_qna_123",
                "message": "What is machine learning?",
                "feature": "qna",
                "user_id": "test_user",
                "auto_save": True
            }
    try:
            # Mock context object
            class MockContext:
                function_name = "test_chat_lambda"
                memory_limit_in_mb = 512
                remaining_time_in_millis = lambda: 30000
            
            # Call lambda handler
            result = lambda_handler(test_case['event'], MockContext())
            
            # Parse and display results
            if result['statusCode'] == 200:
                body = json.loads(result['body'])
                print(f"Status: {result['statusCode']}")
                print(f"Feature: {body.get('feature', 'None')}")
                print(f"Chat ID: {body['chat_id']}")
                print(f"Total Messages: {body['message_count']}")
                print(f"Session Exchanges: {body['session_exchanges']}")
                print(f"Saved Exchanges: {body['saved_exchanges']}")
                print(f"Response Preview: {body['response'][:200]}...")
            else:
                print(f"Status: {result['statusCode']}")
                print(f"Error: {result['body']}")
                
    except Exception as e:
        print(f"Test failed: {str(e)}")




if __name__ == "__main__":
    # Run local tests
    test_lambda_locally()