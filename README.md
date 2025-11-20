Repo with the Backend structure of a Typical GenAI Rag App using AWS Bedrock (or any Framework)

# Repo Structure  
bedrock_genai/  
├── app/  
│   ├── __init__.py  
│   │  
│   ├── 🔧 models/  
│   │   ├── __init__.py  
│   │   ├── bedrock_client.py      # AWS Bedrock integration  
│   │   ├── dynamodb.py            # DynamoDB data layer  
│   │   ├── s3.py                  # S3 storage operations  
│   │   ├── user.py                # User management  
│   │   └── session.py             # Session handling  
│   │  
│   ├── services/  
│   │   ├── __init__.py  
│   │   ├── chat_engine.py         # Chat functionality engine  
│   │   ├── document_processing.py # Document ingestion & processing  
│   │   ├── retrieval.py           # Information retrieval service  
│   │   ├── feature.py             # Feature management  
│   │   └── history.py             # Chat history management  
│   │  
│   ├── utils/  
│   │   ├── __init__.py  
│   │   ├── util.py                # Logger, constants  
│   │   └── helpers.py             # Generic helper functions  
│   │  
│   ├── handlers/  
│   │   ├── __init__.py  
│   │   ├── chat_handler.py        # Chat request handling  
│   │   └── ingestion_handler.py   # Document ingestion handling  
│   │  
│   ├── prompts/  
│   │   └── system_prompts.json    # System prompt templates  
│   │  
│   ├── testdocs/  
│   │   ├── sample_doc.pdf  
│   │   ├── test.pdf  
│   │   ├── test.csv  
│   │   └── test.png  
│   │  
│   └── tests/  
│       ├── __init__.py  
│       ├── test_chat.py               # Chat functionality tests  
│       ├── test_document_processing.py # Document processing tests  
│       ├── test_feature.py            # Feature tests  
│       ├── test_history.py            # History management tests  
│       └── test_image.py              # Image processing tests  
│  
├── infrastructure/  
│   └── cloudFormation.yaml        # AWS CloudFormation templates  
│  
├── scripts/  
│   ├── run_local.py               # Local development server  
│   ├── invoke_core_lambda.py      # Lambda function invoker  
│   └── cognito_lambda.py          # Cognito authentication scripts  
│  
├── requirements.txt           # Python dependencies  
├── .gitignore                # Git ignore patterns  
├── .env.example              # Environment variables template  
└── README.md                 # Project documentation  


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
