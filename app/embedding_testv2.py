import os
import json
import base64
import pandas as pd
from pathlib import Path
import fitz  # PyMuPDF library
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bedrock_client import invoke_claude, invoke_claude_messages, bedrock
import re
import time  # ADDED: For timing measurements

#Async library imports
import asyncio
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_experimental.text_splitter import SemanticChunker

# Simple embedding storage using Titan Embeddings
simple_vector_store = []
executor = ThreadPoolExecutor(max_workers=4)

# ADDED: Timing storage for analytics
timing_logs = {
    'file_processing': [],
    'chunking': [],
    'embedding': [],
    'retrieval': []
}

def log_timing(phase: str, duration: float, details: str = ""):
    """Log timing information for each phase"""
    timing_logs[phase].append({
        'duration': duration,
        'details': details,
        'timestamp': time.time()
    })

def print_timing_summary():
    """Print a summary of all timing measurements"""
    print("\n" + "="*60)
    print("⏱️  PERFORMANCE TIMING SUMMARY")
    print("="*60)
    
    for phase, logs in timing_logs.items():
        if logs:
            total_time = sum(log['duration'] for log in logs)
            avg_time = total_time / len(logs)
            print(f"\n📊 {phase.upper().replace('_', ' ')}:")
            print(f"   Total Time: {total_time:.3f}s")
            print(f"   Average Time: {avg_time:.3f}s")
            print(f"   Operations: {len(logs)}")
            
            for i, log in enumerate(logs):
                if log['details']:
                    print(f"   └─ Operation {i+1}: {log['duration']:.3f}s - {log['details']}")
    
    print("\n" + "="*60)


def get_titan_embedding(text: str) -> list:
    """Get embedding using Bedrock Titan Embeddings V2"""
    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({
                "inputText": text
            })
        )
        response_body = json.loads(response['body'].read())
        return response_body['embedding']
    except Exception as e:
        print(f"Error getting Titan V2 embedding: {e}")
        return None

async def get_titan_embedding_async(text: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_titan_embedding, text)



def read_pdf(file_path: str) -> str:
    """
    Extracts and preprocesses text from a PDF using PyMuPDF.
    """
    start_time = time.time()
    
    try:
        # Use context manager for better resource handling
        with fitz.open(file_path) as doc:
            full_text = []
            page_count = len(doc)

            # Simple pattern for common headers/footers like page numbers
            page_pattern = re.compile(r'Page \d+ of \d+')
            
            for page in doc:
                page_text = page.get_text()

                if page_text:
                    # 1. Normalize whitespace
                    page_text = re.sub(r'\s+', ' ', page_text).strip()

                    # 2. Convert to lowercase
                    page_text = page_text.lower()
                    
                    # 3. Remove headers/footers (basic example)
                    page_text = re.sub(page_pattern, '', page_text)

                    full_text.append(page_text)

        processed_text = ' '.join(full_text)
        
        # Log timing
        duration = time.time() - start_time
        log_timing('file_processing', duration, f"PDF: {len(processed_text)} chars, {page_count} pages")
        
        print(f"📄 PDF processed: {len(processed_text)} characters extracted and preprocessed")
        print(f"⏱️  File processing took: {duration:.3f} seconds")
        return processed_text
        
    except Exception as e:
        duration = time.time() - start_time
        log_timing('file_processing', duration, f"PDF Error: {str(e)}")
        print(f"Error reading PDF with PyMuPDF: {e}")
        return ""

def encode_image(file_path: str) -> str:
    """Encode image to base64"""
    start_time = time.time()  # ADDED: Start timing
    
    try:
        with open(file_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
        # ADDED: Log timing
        duration = time.time() - start_time
        log_timing('file_processing', duration, f"Image: {len(encoded)} chars encoded")
        
        print(f"🖼️  Image encoded: {file_path}")
        print(f"⏱️  Image encoding took: {duration:.3f} seconds")
        return encoded
    except Exception as e:
        duration = time.time() - start_time
        log_timing('file_processing', duration, f"Image Error: {str(e)}")
        print(f"Error encoding image: {e}")
        return ""

def read_csv(file_path: str) -> str:
    """Convert CSV to string"""
    start_time = time.time()  # ADDED: Start timing
    
    try:
        df = pd.read_csv(file_path)
        csv_string = df.to_string(index=False)
        
        # ADDED: Log timing
        duration = time.time() - start_time
        log_timing('file_processing', duration, f"CSV: {df.shape[0]}×{df.shape[1]} -> {len(csv_string)} chars")
        
        print(f"📊 CSV processed: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"⏱️  CSV processing took: {duration:.3f} seconds")
        return csv_string
    except Exception as e:
        duration = time.time() - start_time
        log_timing('file_processing', duration, f"CSV Error: {str(e)}")
        print(f"Error reading CSV: {e}")
        return ""

def store_embedding(content: str, file_type: str, source: str):
    """Store content with Titan V2 embedding, chunking if necessary"""
    chunking_start = time.time()  # ADDED: Start chunking timing

    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split the document into chunks
    chunks = text_splitter.split_text(content)
    
    # ADDED: Log chunking timing
    chunking_duration = time.time() - chunking_start
    log_timing('chunking', chunking_duration, f"{len(chunks)} chunks from {len(content)} chars")
    
    print(f"✂️  Content processed. Created {len(chunks)} chunks.")
    print(f"⏱️  Chunking took: {chunking_duration:.3f} seconds")

    # Run async embedding locally
    asyncio.run(store_embeddings_async(chunks, file_type, source))


async def store_embeddings_async(content: str, file_type: str, source: str,
                                 chunk_size=3000, chunk_overlap=500,
                                 separators=None):
    embedding_start = time.time()  # ADDED: Start embedding timing
    
    separators = separators or ["\n\n", "\n", ". ", " ", ""]

    # Split content into chunks inside the async function
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators
    )
    chunks = text_splitter.split_text(content)

    # Run embeddings in parallel
    print(f"🔄 Creating {len(chunks)} embeddings in parallel...")
    embeddings = await asyncio.gather(*[get_titan_embedding_async(c) for c in chunks])

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if embedding is None:
            print(f"Failed to create embedding for chunk {i+1}")
            continue

        embedding_data = {
            'content': chunk,
            'embedding': embedding,
            'file_type': file_type,
            'source': f"{source} (chunk {i+1}/{len(chunks)})",
            'id': len(simple_vector_store)
        }
        simple_vector_store.append(embedding_data)

    # ADDED: Log embedding timing
    embedding_duration = time.time() - embedding_start
    log_timing('embedding', embedding_duration, f"{len(chunks)} embeddings for {file_type}")
    
    print(f"✅ Successfully stored {len(chunks)} Titan V2 embeddings for {file_type}: {source}")
    print(f"⏱️  Embedding generation took: {embedding_duration:.3f} seconds")

async def store_embeddings_semantic_async(
    content: str,
    file_type: str,
    source: str,
    buffer_size: int = 1,
    breakpoint_threshold_type: str = 'percentile',
    breakpoint_threshold_amount: float | None = None,
    number_of_chunks: int | None = None,
    sentence_split_regex: str = r'(?<=[.?!])\s+',
    min_chunk_size: int | None = 500
):
    """
    Store content with Titan V2 embeddings using LangChain's SemanticChunker.
    Embeddings are created in parallel using asyncio.
    """
    chunking_start = time.time()  # ADDED: Start chunking timing

    # Initialize embeddings for semantic chunking
    embeddings = BedrockEmbeddings(model_name="amazon.titan-embed-text-v2:0")

    # Initialize SemanticChunker with defaults
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        buffer_size=buffer_size,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        number_of_chunks=number_of_chunks,
        sentence_split_regex=sentence_split_regex,
        min_chunk_size=min_chunk_size
    )

    # Split text semantically
    chunks = semantic_chunker.split_text(content)
    
    # ADDED: Log chunking timing
    chunking_duration = time.time() - chunking_start
    log_timing('chunking', chunking_duration, f"Semantic: {len(chunks)} chunks")
    print(f"✂️  Semantic chunking created {len(chunks)} chunks.")
    print(f"⏱️  Semantic chunking took: {chunking_duration:.3f} seconds")

    # Compute embeddings in parallel
    embedding_start = time.time()  # ADDED: Start embedding timing
    embeddings_results = await asyncio.gather(
        *[get_titan_embedding_async(c) for c in chunks]
    )

    # Store in vector store
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_results)):
        if embedding is None:
            print(f"Failed to create embedding for chunk {i+1}")
            continue

        embedding_data = {
            'content': chunk,
            'embedding': embedding,
            'file_type': file_type,
            'source': f"{source} (chunk {i+1}/{len(chunks)})",
            'id': len(simple_vector_store)
        }
        simple_vector_store.append(embedding_data)

    # ADDED: Log embedding timing
    embedding_duration = time.time() - embedding_start
    log_timing('embedding', embedding_duration, f"Semantic: {len(chunks)} embeddings")
    
    print(f"✅ Successfully stored {len(chunks)} Titan V2 embeddings for {file_type} using SemanticChunker: {source}")
    print(f"⏱️  Semantic embedding took: {embedding_duration:.3f} seconds")

def search_content(query: str, top_k: int = 3) -> str:
    """Semantic search using Titan embeddings"""
    retrieval_start = time.time()  # ADDED: Start retrieval timing
    
    if not simple_vector_store:
        return "No content stored yet."
    
    print(f"🔍 Searching through {len(simple_vector_store)} chunks...")
    
    # Get query embedding using Titan
    query_embedding = get_titan_embedding(query)
    if query_embedding is None:
        return "Error creating query embedding."
    
    # Calculate similarities
    similarities = []
    for item in simple_vector_store:
        similarity = cosine_similarity([query_embedding], [item['embedding']])[0][0]
        similarities.append({
            'content': item['content'],
            'similarity': similarity,
            'source': item['source'],
            'file_type': item['file_type']
        })
    
    # Sort by similarity and get top results
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    relevant_content = []
    for result in similarities[:top_k]:
        relevant_content.append(f"From {result['source']} ({result['file_type']}):\n{result['content'][:2000]}...")
    
    # ADDED: Log retrieval timing
    retrieval_duration = time.time() - retrieval_start
    log_timing('retrieval', retrieval_duration, f"Top {top_k} from {len(similarities)} chunks, best match: {similarities[0]['similarity']:.3f}")
    
    print(f"🎯 Retrieved top {top_k} matches (best similarity: {similarities[0]['similarity']:.3f})")
    print(f"⏱️  Retrieval took: {retrieval_duration:.3f} seconds")
    
    return "\n\n---\n\n".join(relevant_content)

def create_rag_prompt(user_question: str, context: str) -> str:
    """Create a simple RAG prompt template"""
    prompt = f"""Based on the following context information, please answer the user's question.

Context:
{context}

Question: {user_question}

Please provide a helpful answer based on the context provided. If the context doesn't contain relevant information, please say so."""
    
    return prompt


def create_rag_prompt_langchain(user_question: str, context: str, chat_history: list = None) -> list:
    """
    Creates a production-ready RAG prompt using LangChain's ChatPromptTemplate,
    with an optional chat history.
    """
    # System message with context
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "You are a helpful AI assistant. Use the following pieces of context to answer the user's question. "
                "If you don't know the answer, just say that you don't know. Keep your answer concise and do not make up an answer.\n\n"
                f"Context: {context}"
            )}]
        }
    ]

    # Add previous chat history if present
    if chat_history:
        for msg in chat_history:
            role = "user" if msg.get("role") == "human" else "assistant"
            messages.append({
                "role": role,
                "content": [{"type": "text", "text": msg.get("content", "")}]
            })

    # Add current user question
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_question}]
    })

    return messages



def process_file(file_path: str):
    """Process different file types"""
    print(f"\n🔄 Processing: {file_path}")
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    if file_path.suffix.lower() == '.pdf':
        content = read_pdf(str(file_path))
        if content:
            asyncio.run(store_embeddings_async(content, 'pdf', str(file_path)))
    
    elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        encoded_image = encode_image(str(file_path))
        if encoded_image:
            # For images, we store a placeholder text for search
            image_text = f"Image file: {file_path.name}"
            store_embedding(image_text, 'image', str(file_path))
            # Store the base64 encoding separately if needed
            simple_vector_store[-1]['image_data'] = encoded_image
    
    elif file_path.suffix.lower() == '.csv':
        content = read_csv(str(file_path))
        if content:
            store_embedding(content, 'csv', str(file_path))
    
    else:
        print(f"❌ Unsupported file type: {file_path.suffix}")

def rag_query_with_langchain(question: str) -> str:
    """Alternative RAG query using LangChain message structure"""
    print(f"\n🤖 Processing query: '{question}'")
    
    # Search for relevant content
    context = search_content(question)
    
    # Create messages using LangChain structure
    messages = create_rag_prompt_langchain(question, context)
    
    # Convert to the format expected by bedrock.converse()
    bedrock_messages = []
    for msg in messages:
        if msg["role"] == "system":
            # Bedrock converse doesn't use system role, so we'll prepend to first user message
            continue
        elif msg["role"] == "user":
            bedrock_messages.append({
                "role": "user",
                "content": [{"text": msg["content"][0]["text"]}]
            })
        elif msg["role"] == "assistant":
            bedrock_messages.append({
                "role": "assistant", 
                "content": [{"text": msg["content"][0]["text"]}]
            })
    
    # Add system context to the first user message
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"][0]["text"]
        if bedrock_messages and bedrock_messages[0]["role"] == "user":
            bedrock_messages[0]["content"][0]["text"] = system_content + "\n\n" + bedrock_messages[0]["content"][0]["text"]
    
    print(f"💬 Sending query to Claude...")
    response_start = time.time()
    
    # Use the modified invoke function
    response = invoke_claude_messages(bedrock_messages)
    
    response_duration = time.time() - response_start
    print(f"⏱️  Claude response took: {response_duration:.3f} seconds")
    
    return response

# Test function
def test_rag_system():
    """Simple test of the RAG system"""
    total_start = time.time()  # ADDED: Total system timing
    
    print("="*60)
    print("🚀 BEDROCK RAG POC TEST")
    print("="*60)
    
    # Example usage (you'll need to provide actual file paths)
    test_files = [
        r"C:\Users\admin\Downloads\bedrock-testing\app\mcdonalds-nutrition-facts.pdf", 
        #"sample.jpg", 
        #"sample.csv" 
    ]
    
    # Process files
    for file_path in test_files:
        if os.path.exists(file_path):
            process_file(file_path)
        else:
            print(f"❌ File not found (skipping): {file_path}")
    
    # Show what's stored
    print(f"\n📦 Stored {len(simple_vector_store)} items in vector store")

    # Test single specific query
    if simple_vector_store:
        print("\n" + "="*60)
        print("🧠 TESTING RAG QUERY")
        print("="*60)
        question = "What is the document about?"
        print(f"\n❓ Q: {question}")
        answer = rag_query_with_langchain(question)
        print(f"\n💡 A: {answer}")
    else:
        print("\n❌ No files processed - add some test files to try the RAG system")
    
    # ADDED: Total system timing
    total_duration = time.time() - total_start
    print(f"\n⏱️  TOTAL SYSTEM TIME: {total_duration:.3f} seconds")
    
    # ADDED: Print timing summary
    print_timing_summary()

if __name__ == "__main__":
    test_rag_system()