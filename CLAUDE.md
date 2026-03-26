# CLAUDE.md — AI Assistant Guide for bedrock-genai

## Project Overview

This is a **Python-based Retrieval-Augmented Generation (RAG) backend** built on AWS Bedrock. It provides:
- Multi-turn chat sessions with Claude via the Bedrock Converse API
- Document ingestion and semantic chunking (PDF, CSV, images)
- In-memory vector store with Titan Embeddings for RAG queries
- AWS Lambda entrypoint for serverless deployment
- Feature-specific system prompts (Q&A, document comparison, search, code review)

---

## Repository Structure

```
bedrock-genai/
├── app/
│   ├── models/
│   │   ├── bedrock_client.py      # AWS Bedrock client + Claude invocation functions (IMPLEMENTED)
│   │   ├── dynamodb.py            # DynamoDB layer (STUB — not implemented)
│   │   ├── s3.py                  # S3 operations (STUB)
│   │   ├── user.py                # User management (STUB)
│   │   └── session.py             # Session handling (STUB)
│   ├── services/
│   │   ├── chat_engine.py         # Chat session management, DynamoDB persistence (IMPLEMENTED)
│   │   ├── document_processing.py # RAG pipeline: PDF/CSV/image ingestion, embeddings (IMPLEMENTED)
│   │   ├── retrieval.py           # Retrieval service (STUB)
│   │   ├── feature.py             # Feature management (STUB)
│   │   └── history.py             # Chat history (STUB)
│   ├── handlers/
│   │   ├── ingestion_handler.py   # Lambda-compatible document ingestion handler (IMPLEMENTED)
│   │   └── chat_handler.py        # Chat request handler (STUB)
│   ├── utils/
│   │   ├── util.py                # Logger, model ID, MODEL_CONFIG constants
│   │   └── helpers.py             # Generic helpers (STUB)
│   ├── prompts/
│   │   └── system_prompts.json    # Feature-specific system prompt templates
│   ├── tests/
│   │   ├── test_chat.py           # Chat engine tests + interactive REPL
│   │   ├── test_image.py          # Multimodal image invocation tests
│   │   ├── test_converse.py       # Bedrock Converse API tests
│   │   ├── test_document_processing.py  # (STUB)
│   │   ├── test_feature.py        # (STUB)
│   │   └── test_history.py        # (STUB)
│   └── testdocs/
│       ├── sample_doc.pdf, test.pdf, test.csv, test.png
├── infrastructure/
│   └── cloudFormation.yaml        # CloudFormation template (STUB — empty)
├── scripts/
│   ├── core_lambda.py             # AWS Lambda handler entrypoint
│   ├── run_local.py               # Local dev server (STUB)
│   ├── invoke_core_lambda.py      # Lambda invoker script
│   └── cognito_lambda.py          # Cognito auth (STUB)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Development Setup

### Prerequisites
- Python 3.10+
- AWS account with Bedrock access enabled
- IAM credentials with `bedrock-runtime:*`, `bedrock:InvokeModel` permissions
- Optional: DynamoDB table and S3 bucket for persistence

### Installation

```bash
git clone <repo-url>
cd bedrock-genai
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then edit .env with your credentials
```

### Environment Variables (`.env`)

```
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1              # bedrock_client.py defaults to us-east-2 if not set
MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0
S3_BUCKET=my-bucket
DYNAMO_TABLE=my-table
```

---

## Running the Code

All modules must be run from the project root as Python modules (not as scripts), so that `app.*` imports resolve correctly.

```bash
# Run document processing / RAG demo
python -m app.services.document_processing

# Run chat tests (quick mode)
python -m app.tests.test_chat --mode quick

# Interactive chat REPL
python -m app.tests.test_chat --mode chat --prompt qna

# Both quick test + REPL
python -m app.tests.test_chat --mode both --prompt docComparison

# Test multimodal image invocation
python -m app.tests.test_image

# Test Bedrock Converse API directly
python -m app.tests.test_converse

# Invoke bedrock_client standalone
python -m app.models.bedrock_client
```

---

## Key Architecture Patterns

### Layered Architecture
```
Handlers  →  Services  →  Models
(Lambda/API)   (business logic)   (AWS clients)
```

### Bedrock Converse API Message Format
All Claude invocations use the Bedrock `converse()` API with this structure:
```python
messages = [
    {"role": "user", "content": [{"text": "your prompt"}]},
    {"role": "assistant", "content": [{"text": "prior response"}]},
]
response_text = response["output"]["message"]["content"][0]["text"]
```

### Two Invocation Functions (`app/models/bedrock_client.py`)
- `invoke_claude(prompt: str, ...)` — wraps a single string into a user message
- `invoke_claude_messages(messages: list, ...)` — accepts a pre-built message list (for multi-turn)

### Model Configuration (`app/utils/util.py`)
The active model is set by editing `modelId` in `util.py`:
```python
modelId = "us.anthropic.claude-3-5-haiku-20241022-v1:0"   # active
# modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"  # commented out
modelType = "sonnet" if "sonnet" in modelId else "haiku"
```

`MODEL_CONFIG` controls per-model limits:
| Model  | max_tokens | max_rows (CSV) |
|--------|-----------|----------------|
| haiku  | 4096      | 300            |
| sonnet | 200000    | None (no limit)|

### Chat Engine (`app/services/chat_engine.py`)
- `ChatEngine(chat_id, system_prompt, mode, use_dynamo, metadata)` — one instance per session
- **mode="api"**: returns structured JSON responses
- **mode="chat"**: returns conversational text
- Uses `DynamoChatStorage` (imported from `dynamo_chat_storage` — external dependency) when `use_dynamo=True`; falls back to in-memory list silently if not available
- `session_exchanges` tracks unsaved turns; `conversation_history` holds DynamoDB-loaded history

### Document Processing & RAG (`app/services/document_processing.py`)
- Global `simple_vector_store: list` — in-memory vector store; not persistent across runs
- `SemanticChunkingTitanEmbeddings` — LangChain `Embeddings` subclass wrapping Amazon Titan `amazon.titan-embed-text-v2:0`
- Similarity search uses `cosine_similarity` from scikit-learn
- Embeddings are stored asynchronously via `ThreadPoolExecutor(max_workers=10)`
- PDF preprocessing: lowercase, whitespace normalization, header/footer removal, special char stripping
- CSV row limits enforced per model type (see MODEL_CONFIG above)
- Images are passed as raw bytes (not base64 strings) to `bedrock.converse()`
- **Haiku does not support images** — validated in `process_file()`

### System Prompts (`app/prompts/system_prompts.json`)
Four feature keys: `qna`, `docComparison`, `search`, `codeReview`. Loaded and cached by `chat_engine.py`.

---

## Testing

Tests are in `app/tests/`. Use the CLI flags in `test_chat.py`:

```bash
# Available --prompt values: qna | docComparison | search | codeReview
# Available --mode values:   quick | chat | both
python -m app.tests.test_chat --mode quick --prompt qna
```

Test documents are in `app/testdocs/` (PDF, CSV, PNG).

There is no pytest configuration file. Stub test files exist but contain no tests yet.

---

## Logging

Logging is configured in `util.py` at module import time:
- **File**: `rag_system.log` (written in the project root, relative to CWD)
- **Console**: stdout via `StreamHandler`
- **Level**: INFO
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

Import the shared logger: `from app.utils.util import logger`

---

## AWS Lambda Deployment

Entry point: `scripts/core_lambda.py`
```python
def lambda_handler(event, context):
    return ingestion_handler.handle_ingestion(event)
```

Expected event payload for ingestion:
```json
{
  "s3_bucket": "my-bucket",
  "s3_key": "path/to/document.pdf",
  "user_prompt": "Summarize this document"
}
```

---

## What Is Not Yet Implemented

The following files are empty stubs — do not assume functionality exists:
- `app/models/dynamodb.py`, `s3.py`, `user.py`, `session.py`
- `app/services/retrieval.py`, `feature.py`, `history.py`
- `app/handlers/chat_handler.py`
- `app/utils/helpers.py`
- `scripts/run_local.py`, `cognito_lambda.py`
- `infrastructure/cloudFormation.yaml`
- Most test files (`test_document_processing.py`, `test_feature.py`, `test_history.py`)

`DynamoChatStorage` (imported in `chat_engine.py`) is also not in this repo — it's expected as a separate external module or Lambda layer.

---

## Conventions to Follow

1. **Always run as modules from project root**: `python -m app.<path>` — never `python app/services/script.py`
2. **Import the shared logger**: use `from app.utils.util import logger` instead of creating new loggers
3. **Model ID changes go in `util.py`** — `modelId` is imported everywhere else
4. **Message structure for Bedrock**: always use `[{"role": ..., "content": [{"text": ...}]}]` format
5. **Response extraction**: always use `response["output"]["message"]["content"][0]["text"]`
6. **Error handling**: wrap all AWS calls in try/except and log with `logger.error()`
7. **Type hints**: use `List`, `Dict`, `Optional`, `Literal` from `typing` (project targets Python 3.10)
8. **Do not commit `.env`** — it is gitignored; use `.env.example` as template
9. **Async embedding**: use `ThreadPoolExecutor` for concurrent embedding tasks, not raw threads
10. **CSV row limits**: always call `get_max_rows(model_type)` from `util.py` — do not hardcode row limits
