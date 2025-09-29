from app.handlers import ingestion_handler

## Add more logic like userId, sessionId, etc. as needed
def lambda_handler(event, context):
    """
    AWS Lambda entrypoint for document ingestion + user prompt.
    """
    try:
        return ingestion_handler.handle_ingestion(event)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": {"error": str(e)}
        }
