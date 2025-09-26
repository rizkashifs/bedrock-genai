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
    try:
        # Open the PDF document
        doc = fitz.open(file_path)
        full_text = []

        # Simple pattern for common headers/footers like page numbers
        # This will need to be customized for your specific documents
        page_pattern = re.compile(r'Page \d+ of \d+')
        
        for page in doc:
            page_text = page.get_text()

            if page_text:
                # 1. Normalize whitespace
                # Replace multiple spaces, tabs, and newlines with a single space
                page_text = re.sub(r'\s+', ' ', page_text).strip()

                # 2. Convert to lowercase
                page_text = page_text.lower()
                
                # 3. Remove headers/footers (basic example)
                page_text = re.sub(page_pattern, '', page_text)

                full_text.append(page_text)
        
        doc.close()

        processed_text = ' '.join(full_text)
        
        print(f"PDF processed: {len(processed_text)} characters extracted and preprocessed")
        return processed_text
    except Exception as e:
        print(f"Error reading PDF with PyMuPDF: {e}")
        return ""

def encode_image(file_path: str) -> str:
    """Encode image to base64"""
    try:
        with open(file_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        print(f"Image encoded: {file_path}")
        return encoded
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""

def read_csv(file_path: str) -> str:
    """Convert CSV to string"""
    try:
        df = pd.read_csv(file_path)
        csv_string = df.to_string(index=False)
        print(f"CSV processed: {df.shape[0]} rows, {df.shape[1]} columns")
        return csv_string
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return ""

def store_embedding(content: str, file_type: str, source: str):
    """Store content with Titan V2 embedding, chunking if necessary"""

    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split the document into chunks
    chunks = text_splitter.split_text(content)
    print(f"Content processed. Created {len(chunks)} chunks.")

    # Run async embedding locally
    asyncio.run(store_embeddings_async(chunks, file_type, source))


async def store_embeddings_async(content: str, file_type: str, source: str,
                                 chunk_size=3000, chunk_overlap=500,
                                 separators=None):
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

    print(f"Successfully stored {len(chunks)} Titan V2 embeddings for {file_type}: {source}")

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

    Args:
        content (str): Full text to chunk and embed.
        file_type (str): File type (e.g., 'pdf', 'csv', 'image').
        source (str): Source identifier (e.g., file path).
        buffer_size (int): Number of chunks to overlap (default=1).
        breakpoint_threshold_type (str): 'percentile', 'standard_deviation', 'interquartile', 'gradient'.
        breakpoint_threshold_amount (float | None): Threshold value for breakpoints.
        number_of_chunks (int | None): Max number of chunks to generate.
        sentence_split_regex (str): Regex to split sentences for semantic chunking.
        min_chunk_size (int | None): Minimum chunk size in characters.
    """

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
    print(f"Semantic chunking created {len(chunks)} chunks.")

    # Compute embeddings in parallel
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

    print(f"Successfully stored {len(chunks)} Titan V2 embeddings for {file_type} using SemanticChunker: {source}")

def search_content(query: str, top_k: int = 3) -> str:
    """Semantic search using Titan embeddings"""
    if not simple_vector_store:
        return "No content stored yet."
    
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
    
    # Print the highest scoring chunk
    if similarities:
        best_match = similarities[0]
        #print(f"\n=== HIGHEST SIMILARITY CHUNK ===")
        #print(f"Source: {best_match['source']}")
        #print(f"Similarity: {best_match['similarity']:.3f}")
        #print(f"Content ({len(best_match['content'])} chars):")
        #print("-" * 80)
        #print(best_match['content'])
        #print("-" * 80)
    
    relevant_content = []
    for result in similarities[:top_k]:
        #print(f"Match: {result['source']} (similarity: {result['similarity']:.3f})")
        relevant_content.append(f"From {result['source']} ({result['file_type']}):\n{result['content'][:2000]}...")
    
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
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
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
        print(f"Unsupported file type: {file_path.suffix}")

def rag_query_with_langchain(question: str) -> str:
    """Alternative RAG query using LangChain message structure"""
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
    
    # Debug print
    #print(f"\n=== BEDROCK MESSAGES ===")
    #print(json.dumps(bedrock_messages, indent=2))
    #print("=" * 40)
    
    # Use the modified invoke function
    return invoke_claude_messages(bedrock_messages)

# Test function
def test_rag_system():
    """Simple test of the RAG system"""
    print("=== Bedrock RAG POC Test ===\n")
    
    # Example usage (you'll need to provide actual file paths)
    test_files = [
        r"C:\Users\admin\Downloads\bedrock-testing\app\mcdonalds-nutrition-facts.pdf", 
        #"sample.jpg", 
        #"sample.csv" 
    ]
    
    # Process files
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\nProcessing: {file_path}")
            process_file(file_path)
        else:
            print(f"File not found (skipping): {file_path}")
    
    # Show what's stored
    print(f"\nStored {len(simple_vector_store)} items in vector store")

    #for item in simple_vector_store:
    #    print(f"- {item['file_type']}: {item['source']}")
    
    # Test single specific query
    if simple_vector_store:
        print("\n=== Testing RAG Query ===")
        question = "What is the document about?"
        print(f"\nQ: {question}")
        answer = rag_query_with_langchain(question)
        print(f"\nA: {answer}")
    else:
        print("\nNo files processed - add some test files to try the RAG system")

if __name__ == "__main__":
    test_rag_system()