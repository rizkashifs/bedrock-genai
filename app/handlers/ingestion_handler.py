# app/handlers/ingestion_handler.py

## To be updated with actual ingestion logic
from app.services import document_processing

def handle_ingestion(event):
    """
    Handler for document ingestion + QA.
    Expects: {
        "s3_bucket": "bucket-name",
        "s3_key": "path/to/file.pdf",
        "user_prompt": "Summarize this document"
    }
    """
    bucket = event.get("s3_bucket")
    key = event.get("s3_key")
    user_prompt = event.get("user_prompt")

    if not bucket or not key:
        raise ValueError("Missing required parameters: s3_bucket and s3_key")
    if not user_prompt:
        raise ValueError("Missing required parameter: user_prompt")

    # Process the document (extract text / embeddings)
    doc_context = document_processing.process_document(bucket, key)

    # Very simple placeholder logic (replace with RAG/LLM call later)
    answer = f"Processed {key}. User asked: '{user_prompt}'. Context: {doc_context[:200]}..."

    return {
        "statusCode": 200,
        "body": {
            "answer": answer
        }
    }