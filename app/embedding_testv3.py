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
import logging
import time
import numpy as np
#Async library imports
import asyncio
from concurrent.futures import ThreadPoolExecutor
# Class imports
from pydantic import BaseModel, Field
from typing import List, Any

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simple embedding storage using Titan Embeddings
simple_vector_store = []
executor = ThreadPoolExecutor(max_workers=10)


class SemanticChunkingTitanEmbeddings(BaseModel, Embeddings):
    """
    Custom LangChain Embeddings wrapper for SemanticChunker, relying on a 
    pre-existing global 'bedrock' Boto3 client for synchronous Bedrock API calls.
    """
    
    model_id: str = Field(default="amazon.titan-embed-text-v2:0")

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single piece of text using the global 'bedrock' client.
        """
        
        try:
            response = bedrock.invoke_model(  # <--- Using the global client
                modelId=self.model_id,
                body=json.dumps({
                    "inputText": text
                })
            )
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
        except Exception as e:
            logger.error(f"Error getting Titan V2 embedding for semantic chunker: {e}")
            return None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents by calling embed_query for each text.
        """
        return [self.embed_query(text) for text in texts if self.embed_query(text) is not None]


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
        logger.error(f"Error getting Titan V2 embedding: {e}")
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
        
        logger.info(f"PDF processed: {len(processed_text)} characters extracted and preprocessed")
        return processed_text
    except Exception as e:
        logger.error(f"Error reading PDF with PyMuPDF: {e}")
        return ""

def encode_image(file_path: str) -> str:
    """Encode image to base64"""
    try:
        with open(file_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        logger.info(f"Image encoded: {file_path}")
        return encoded
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return ""

def read_csv(file_path: str) -> str:
    """Convert CSV to string"""
    try:
        df = pd.read_csv(file_path)
        csv_string = df.to_string(index=False)
        logger.info(f"CSV processed: {df.shape[0]} rows, {df.shape[1]} columns")
        return csv_string
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return ""

def store_embedding(content: str, file_type: str, source: str):
    """
    Store content with Titan V2 embedding using RecursiveCharacterTextSplitter.
    Handles chunking and launches the async embedding process.
    """   
    # Time monitoring for Chunking
    chunking_start_time = time.time()
    logger.info("Starting chunking process")

    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split the document into chunks
    chunks = text_splitter.split_text(content)
    
    chunking_end_time = time.time()
    chunking_duration = chunking_end_time - chunking_start_time
    logger.info(f"Chunking completed. Created {len(chunks)} chunks in {chunking_duration:.2f} seconds")

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

    # Time monitoring for Embedding
    embedding_start_time = time.time()
    logger.info("Starting embedding process")

    # Run embeddings in parallel
    embeddings = await asyncio.gather(*[get_titan_embedding_async(c) for c in chunks])

    embedding_end_time = time.time()
    embedding_duration = embedding_end_time - embedding_start_time
    logger.info(f"Embedding completed in {embedding_duration:.2f} seconds")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if embedding is None:
            logger.warning(f"Failed to create embedding for chunk {i+1}")
            continue

        embedding_data = {
            'content': chunk,
            'embedding': embedding,
            'file_type': file_type,
            'source': f"{source} (chunk {i+1}/{len(chunks)})",
            'id': len(simple_vector_store)
        }
        simple_vector_store.append(embedding_data)

    logger.info(f"Successfully stored {len(chunks)} Titan V2 embeddings for {file_type}: {source}")

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

    # Time monitoring for Chunking (Semantic)
    chunking_start_time = time.time()
    logger.info("Starting semantic chunking process using custom Titan wrapper")

    # 1. Initialize your custom embeddings wrapper (no need to pass the client)
    embeddings = SemanticChunkingTitanEmbeddings(model_id="amazon.titan-embed-text-v2:0")


    # Debug: Test embedding to ensure BedrockEmbeddings is working
    try:
        test_embedding = embeddings.embed_query("This is a test sentence.")
        logger.info(f"Semantic BedrockEmbeddings test embedding length: {len(test_embedding)}") 
        if not test_embedding:
            logger.error("Semantic BedrockEmbeddings failed to return a test embedding.")
    except Exception as e:
            logger.error(f"Semantic BedrockEmbeddings initialization or test failed: {e}")


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
    try: 
        chunks = semantic_chunker.split_text(content)

    except Exception as e:
        # Capture the specific ValidationException error
        logger.error(f"Semantic chunking FAILED due to API/Size constraint: {e}")
        # Check if the error is the exact size validation error
        if "maxLength" in str(e):
                logger.error("HINT: Input text is too large for the embedding model's API limit (likely 50,000 characters). You must pre-chunk the document.")
        
        # Return an empty list or raise the error further, depending on desired behavior
        return # Return empty list so the rest of the function gracefully exits
    
    chunking_end_time = time.time()
    chunking_duration = chunking_end_time - chunking_start_time
    logger.info(f"Semantic chunking completed. Created {len(chunks)} chunks in {chunking_duration:.2f} seconds")

    # Time monitoring for Embedding
    embedding_start_time = time.time()
    logger.info("Starting embedding process")

    # Compute embeddings in parallel
    embeddings_results = await asyncio.gather(
        *[get_titan_embedding_async(c) for c in chunks]
    )

    embedding_end_time = time.time()
    embedding_duration = embedding_end_time - embedding_start_time
    logger.info(f"Embedding completed in {embedding_duration:.2f} seconds")

    # Store in vector store
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_results)):
        if embedding is None:
            logger.warning(f"Failed to create embedding for chunk {i+1}")
            continue

        embedding_data = {
            'content': chunk,
            'embedding': embedding,
            'file_type': file_type,
            'source': f"{source} (chunk {i+1}/{len(chunks)})",
            'id': len(simple_vector_store)
        }
        simple_vector_store.append(embedding_data)

    logger.info(f"Successfully stored {len(chunks)} Titan V2 embeddings for {file_type} using SemanticChunker: {source}")

def search_content(query: str, top_k: int = 3) -> str:
    """Semantic search using Titan embeddings"""
    # Time monitoring for Retrieval
    retrieval_start_time = time.time()
    logger.info("Starting retrieval process")
    
    if not simple_vector_store:
        logger.warning("No content stored yet")
        return "No content stored yet."
    
    # Get query embedding using Titan
    query_embedding = get_titan_embedding(query)
    if query_embedding is None:
        logger.error("Error creating query embedding")
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
    
    retrieval_end_time = time.time()
    retrieval_duration = retrieval_end_time - retrieval_start_time
    logger.info(f"Retrieval completed in {retrieval_duration:.2f} seconds")
    
    # Print the highest scoring chunk
    if similarities:
        best_match = similarities[0]
        #logger.info(f"\n=== HIGHEST SIMILARITY CHUNK ===")
        #logger.info(f"Source: {best_match['source']}")
        #logger.info(f"Similarity: {best_match['similarity']:.3f}")
        #logger.info(f"Content ({len(best_match['content'])} chars):")
        #logger.info("-" * 80)
        #logger.info(best_match['content'])
        #logger.info("-" * 80)
    
    relevant_content = []
    for result in similarities[:top_k]:
        #logger.info(f"Match: {result['source']} (similarity: {result['similarity']:.3f})")
        relevant_content.append(f"From {result['source']} ({result['file_type']}):\n{result['content'][:2000]}...")
    
    return "\n\n---\n\n".join(relevant_content)

def search_content_numpy(query: str, top_k: int = 3) -> str:
    """Semantic search using Titan embeddings with optimized retrieval"""
    # Time monitoring for Retrieval
    retrieval_start_time = time.time()
    logger.info("Starting retrieval process")
    
    if not simple_vector_store:
        logger.warning("No content stored yet")
        return "No content stored yet."
    
    # Get query embedding using Titan
    query_embedding = get_titan_embedding(query)
    if query_embedding is None:
        logger.error("Error creating query embedding")
        return "Error creating query embedding."
    
    # Vectorized similarity calculation (much faster than sklearn loop)
    embeddings_matrix = np.array([item['embedding'] for item in simple_vector_store])
    query_embedding_array = np.array(query_embedding).reshape(1, -1)
    
    # Calculate all similarities at once using numpy
    similarities_scores = np.dot(embeddings_matrix, query_embedding_array.T).flatten()
    norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_embedding_array)
    similarities_scores = similarities_scores / norms
    
    # Early termination if we find very high similarity matches (>0.95)
    high_similarity_threshold = 0.95
    high_similarity_indices = np.where(similarities_scores > high_similarity_threshold)[0]
    
    if len(high_similarity_indices) > 0:
        logger.info(f"Found {len(high_similarity_indices)} high-similarity matches (>{high_similarity_threshold}), using early termination")
        # Use only high similarity matches, sorted by score
        sorted_indices = high_similarity_indices[np.argsort(-similarities_scores[high_similarity_indices])]
        top_indices = sorted_indices[:min(top_k, len(sorted_indices))]
    else:
        # Get top_k indices without sorting everything (partial sort)
        top_indices = np.argpartition(similarities_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-similarities_scores[top_indices])]
    
    retrieval_end_time = time.time()
    retrieval_duration = retrieval_end_time - retrieval_start_time
    logger.info(f"Retrieval completed in {retrieval_duration:.2f} seconds")
    
    # Build results
    relevant_content = []
    for idx in top_indices:
        item = simple_vector_store[idx]
        similarity = similarities_scores[idx]
        relevant_content.append(f"From {item['source']} ({item['file_type']}):\n{item['content'][:2000]}...")
        logger.info(f"Match: {item['source']} (similarity: {similarity:.3f})")
    
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
        logger.error(f"File not found: {file_path}")
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
        logger.warning(f"Unsupported file type: {file_path.suffix}")

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
    #logger.debug(f"\n=== BEDROCK MESSAGES ===")
    #logger.debug(json.dumps(bedrock_messages, indent=2))
    #logger.debug("=" * 40)
    
    # Use the modified invoke function
    return invoke_claude_messages(bedrock_messages)

# Test function
def test_rag_system():
    """Simple test of the RAG system"""
    logger.info("=== Bedrock RAG POC Test ===\n")
    
    # Example usage (you'll need to provide actual file paths)
    test_files = [
        r"C:\Users\admin\Downloads\bedrock-testing\app\mcdonalds-nutrition-facts.pdf", 
        #"sample.jpg", 
        #"sample.csv" 
    ]
    
    # Process files
    for file_path in test_files:
        if os.path.exists(file_path):
            logger.info(f"\nProcessing: {file_path}")
            process_file(file_path)
        else:
            logger.warning(f"File not found (skipping): {file_path}")
    
    # Show what's stored
    logger.info(f"Stored {len(simple_vector_store)} items in vector store")

    #for item in simple_vector_store:
    #    logger.info(f"- {item['file_type']}: {item['source']}")
    
    # Test single specific query
    if simple_vector_store:
        logger.info("=== Testing RAG Query ===")
        question = "What is the document about?"
        logger.info(f"Q: {question}")
        answer = rag_query_with_langchain(question)
        logger.info(f"A: {answer}")
    else:
        logger.info("No files processed - add some test files to try the RAG system")

if __name__ == "__main__":
    test_rag_system()