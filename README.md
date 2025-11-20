# Bedrock RAG FastAPI Application

A production-ready RAG (Retrieval-Augmented Generation) application built with FastAPI and AWS Bedrock. This application provides a modular architecture for building chat interfaces with document context, history management, and LLM integration.

## Features

- **FastAPI Backend**: High-performance, easy-to-use API framework.
- **AWS Bedrock Integration**: Uses Claude (via Bedrock) for LLM capabilities.
- **RAG Architecture**: Integrated retrieval system for document context.
- **Chat Orchestration**: Manages conversation flow, history, and context injection.
- **Modular Design**: Separate services for history, retrieval, and chat logic.
- **Extensible**: Easy to add new features or swap components (e.g., vector stores, databases).

## Project Structure

```
bedrock_genai/
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── models/                # Data models (Pydantic, Bedrock client)
│   ├── services/              # Core logic (Chat, History, Retrieval)
│   ├── utils/                 # Helper functions
│   └── ...
├── tests/                     # Unit and integration tests
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Prerequisites

- Python 3.10+
- AWS Account with Bedrock access enabled (specifically for Claude models)
- AWS Credentials configured locally (or via IAM roles)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bedrock-testing
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup:**
   Create a `.env` file in the root directory:
   ```env
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   MODEL_ID=anthropic.claude-v2
   ```

## Running the Application

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive API documentation (Swagger UI) is at `http://localhost:8000/docs`.

## Testing

Run the mock tests (no AWS credentials required):
```bash
python tests/test_api_mock.py
```

## API Endpoints

### POST /chat
Send a message to the chat engine.

**Request Body:**
```json
{
  "message": "What is RAG?",
  "chat_id": "optional-uuid",
  "system_prompt": "optional-system-prompt",
  "feature": "qna"
}
```

**Response:**
```json
{
  "response": "RAG stands for...",
  "chat_id": "uuid",
  "history": [...]
}
```

### GET /health
Health check endpoint.
