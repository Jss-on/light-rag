"""
Lightweight LangChain‑based Retrieval‑Augmented Generation (RAG) service
======================================================================
• LLMs: Cohere (embeddings) + Groq (generation)
• Vector store: LanceDB (local directory "./lancedb")
• Framework: FastAPI (with streaming responses by default)
• Data source: .txt meeting transcript files via /ingest endpoint

Run:
  export COHERE_API_KEY="..."  # required for Cohere Embeddings
  export GROQ_API_KEY="..."    # required for Groq Chat
  pip install -r requirements.txt
  uvicorn app:app --reload

Endpoints:
  POST /ingest   – multipart/form‑data {files: List[UploadFile]}  ➜ indexes transcripts
  GET  /query?q= – streams answer tokens as plain text
"""
from __future__ import annotations

import os
import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

# Import dotenv for loading environment variables from .env file
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
import lancedb

# ---------------------------------------------------------------------------
# Logging setup -------------------------------------------------------------
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration --------------------------------------------------------------
# ---------------------------------------------------------------------------
DATA_DIR = Path("./lancedb")          # LanceDB directory (created automatically)
TABLE_NAME = "meeting_transcripts"     # LanceDB table name
CHUNK_SIZE = 1000                      # characters per chunk (adjust as needed)
CHUNK_OVERLAP = 100                    # overlap to preserve context
TOP_K = 4                              # number of chunks to retrieve for each query

# Load environment variables from .env file
load_dotenv()
logger.info("Loaded environment variables from .env file")

# API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not COHERE_API_KEY:
    logger.error("COHERE_API_KEY environment variable not set")
    raise RuntimeError("COHERE_API_KEY environment variable not set")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set")
    raise RuntimeError("GROQ_API_KEY environment variable not set")

# ---------------------------------------------------------------------------
# FastAPI App Setup ---------------------------------------------------------
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Lightweight Meeting‑Transcript RAG Service",
    description="A RAG service for meeting transcripts using LangChain with Cohere and Groq",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Process time: {process_time:.4f}s"
    )
    return response

# ---------------------------------------------------------------------------
# LLM / Embeddings / Vector Store Setup ------------------------------------
# ---------------------------------------------------------------------------
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0", 
    cohere_api_key=COHERE_API_KEY
)
logger.info("Initialized Cohere embeddings model: embed-multilingual-v3.0")

chat_model = ChatGroq(
    model="deepseek-r1-distill-llama-70b", 
    temperature=0.0, 
    api_key=GROQ_API_KEY
)
logger.info("Initialized Groq chat model: deepseek-r1-distill-llama-70b")

# Create or open LanceDB table once at startup
# ---------------------------------------------------------------------------
# Helper Utilities ----------------------------------------------------------
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """Handles document processing operations."""
    
    @staticmethod
    def split_documents(raw_docs: List[Document]) -> List[Document]:
        """Split raw documents into overlapping chunks for semantic search."""
        logger.debug(f"Splitting {len(raw_docs)} documents into chunks")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        
        chunks = splitter.split_documents(raw_docs)
        logger.debug(f"Created {len(chunks)} chunks from {len(raw_docs)} documents")
        return chunks


class RetrievalService:
    """Handles context retrieval and response generation."""
    
    @staticmethod
    def prepare_context(question: str, k: int = TOP_K) -> str:
        """Retrieve k most relevant chunks and build a single context string."""
        assert vector_store is not None
        
        logger.debug(f"Retrieving top {k} chunks for question: {question}")
        start_time = time.time()
        docs = vector_store.similarity_search(question, k=k)
        elapsed = time.time() - start_time
        
        logger.debug(f"Retrieved {len(docs)} chunks in {elapsed:.4f}s")
        return "\n".join(d.page_content for d in docs)
    
    @staticmethod
    async def stream_answer(question: str) -> AsyncGenerator[str, None]:
        """Async generator yielding answer tokens as they stream from Groq LLM."""
        logger.info(f"Generating streaming answer for question: {question}")
        
        # Get context from vector store
        start_time = time.time()
        context = RetrievalService.prepare_context(question)
        retrieval_time = time.time() - start_time
        logger.debug(f"Context retrieval completed in {retrieval_time:.4f}s")
        
        # Prepare prompts for the LLM
        system_prompt = (
            "You are an AI meeting assistant. Answer questions using ONLY the context "
            "provided below. If the answer is not in the context, say you don't know."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # Stream response from the LLM
        token_count = 0
        generation_start = time.time()
        
        # chat_model.stream returns an iterator of ChatCompletionChunk objects
        for chunk in chat_model.stream(system_prompt=system_prompt, user_message=user_message):
            # Each chunk may contain multiple choices; we concatenate their deltas
            for choice in chunk.choices:
                delta = choice.delta  # type: ignore[attr-defined]
                if delta and getattr(delta, "content", None):
                    token_count += 1
                    yield delta.content
                    
        # Ensure final flush / newline
        generation_time = time.time() - generation_start
        logger.info(f"Generated {token_count} tokens in {generation_time:.4f}s")
        yield "\n"


# ---------------------------------------------------------------------------
# API Endpoints -------------------------------------------------------------
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)) -> JSONResponse:  # noqa: B008
    """
    Ingest and index one or more .txt files containing meeting transcripts.
    
    Args:
        files: List of text files to ingest
        
    Returns:
        JSONResponse with status and number of indexed chunks
    
    Raises:
        HTTPException: If no files are provided or non-txt files are included
    """
    # Validate request
    if not files:
        logger.warning("Ingest request received with no files")
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    logger.info(f"Ingesting {len(files)} files")
    start_time = time.time()
    
    # Process each file
    raw_docs: List[Document] = []
    for f in files:
        if not f.filename.lower().endswith(".txt"):
            logger.warning(f"Rejected non-txt file: {f.filename}")
            raise HTTPException(status_code=400, detail="Only .txt files are allowed")
        
        try:
            logger.info(f"Processing file: {f.filename}")
            content = await f.read()
            text = content.decode("utf-8", errors="ignore")
            
            # Add metadata to document
            metadata = {
                "filename": f.filename,
                "ingestion_timestamp": datetime.now().isoformat(),
                "file_size_bytes": len(content)
            }
            
            raw_docs.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            logger.error(f"Error processing file {f.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file {f.filename}: {str(e)}")
    
    # Split documents into chunks
    try:
        split_docs = DocumentProcessor.split_documents(raw_docs)
        
        # Store in vector database
        LanceDB.add_documents(split_docs)
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully indexed {len(split_docs)} chunks from {len(files)} files in {processing_time:.4f}s")
        
        return JSONResponse({
            "status": "ok", 
            "indexed_chunks": len(split_docs),
            "files_processed": len(files),
            "processing_time_seconds": round(processing_time, 4)
        })
    except Exception as e:
        logger.error(f"Error during document processing or indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during document processing: {str(e)}")


@app.get("/query")
async def query(q: str) -> StreamingResponse:
    """
    Retrieve context from LanceDB and stream an answer via Groq LLM.
    
    Args:
        q: Query string
        
    Returns:
        StreamingResponse with tokens from the LLM
    
    Raises:
        HTTPException: If query is empty
    """
    # Validate query
    if not q.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Processing query: {q}")
    
    try:
        # Return streaming response
        return StreamingResponse(
            RetrievalService.stream_answer(q), 
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint to verify service status."""
    logger.debug("Health check request received")
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "embedding_model": "embed-multilingual-v3.0",
            "llm_model": "deepseek-r1-distill-llama-70b"
        }
    })


# ---------------------------------------------------------------------------
# Development convenience: Run with `python app.py` ------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting server in development mode")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
