Repo with the Backend structure of a Typical GenAI Rag App using Bedrock

# Repo Structure
bedrock_genai/
│
├─ app/
│ ├─ init.py
│ ├─ models/
│ │ ├─ init.py
│ │ ├─ bedrock_client.py
│ │ ├─ dynamodb.py
│ │ ├─ s3.py
│ │ ├─ user.py
│ │ └─ session.py
│ │
│ ├─ services/
│ │ ├─ init.py
│ │ ├─ chat_engine.py
│ │ ├─ document_processing.py
│ │ ├─ retrieval.py
│ │ ├─ feature.py
│ │ ├─ history.py
│ │
│ ├─ utils/
│ │ ├─ init.py
│ │ ├─ util.py # logger, constants
│ │ └─ helpers.py # generic helper functions
│ │
│ ├─ handlers/
│ │ ├─ init.py
│ │ ├─ chat_handler.py
│ │ ├─ ingestion_handler.py
│ │
│ ├─ prompts/
│ │ └─ system_prompts.json
│ │
│ ├─ testdocs/
│ │ └─ sample_doc.pdf
│ │ └─ test.pdf
│ │ └─ test.csv
│ │ └─ test.png
│ │
│ └─ tests/
│ ├─ init.py
│ ├─ test_chat.py
│ ├─ test_document_processing.py
│ ├─ test_feature.py
│ ├─ test_history.py
│ └─ test_image.py
│
├─ infrastructure/
│ └─ cloudFormation.yaml
│
├─ scripts/
│ ├─ run_local.py
│ └─ invoke_core_lambda.py
│ └─ cognito_lambda.py
│
├─ requirements.txt
├─ .gitignore
├─ .env.example
└─ README.md


# Document Processing
This file explains how to run the document ingestion/processing module locally.

## Purpose
`app.services.document_processing` ingests documents, chunks them, computes embeddings, and stores metadata/results for the RAG pipeline.

## Prerequisites
- Python 3.10+ (or your project Python)
- Virtual environment (recommended)
- AWS credentials configured if using S3 / DynamoDB / Bedrock

## Setup (one-time)
git clone <your-repo-url>
cd bedrock-testing

python -m venv venv
# mac/linux
source venv/bin/activate
# windows (PowerShell)
venv\Scripts\activate

pip install -r requirements.txt

# Environment variables

Create a .env in the project root (do NOT commit). Example:
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1
MODEL_ID=anthropic.claude-v2
S3_BUCKET=my-bucket
DYNAMO_TABLE=my-table


# Run document processing (always from project root)

Important: run as a module so app package imports resolve correctly.

#From project root (bedrock-testing/)
python -m app.services.document_processing



# Other modules on the way..
