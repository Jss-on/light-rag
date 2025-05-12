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

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
CHUNK_SIZE = 100000                      # characters per chunk (adjust as needed)
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
embedding_model = CohereEmbeddings(
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
vector_store = LanceDB(embedding=embedding_model, uri=DATA_DIR, table_name=TABLE_NAME)
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
    """Handles context retrieval and response generation using LangChain QA chain."""
    
    
    def __init__(self):
        logger.info("Initializing QA chain")
        # Configure retriever with more options for better results
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": TOP_K,
            }
        )
        logger.info(f"Configured retriever with k={TOP_K}, fetch_k={TOP_K * 2}")
        
        # Create a more detailed prompt template with better instructions
        self.prompt = ChatPromptTemplate.from_template(
            """You are an AI meeting assistant that helps extract information from meeting transcripts.

"""
            """CONTEXT INFORMATION:
{context}

"""
            """QUESTION: {question}

"""
            """INSTRUCTIONS:
1. Answer the question based ONLY on the context provided above.
2. If the answer is not in the context, respond with 'I don't have that information in the meeting transcript.'
3. Be concise and to the point, focusing on the specific information requested.
4. If you quote from the transcript, use the exact wording.
5. If multiple people discussed the topic, mention their perspectives.
6. DO NOT use <think> tags in your response.

ANSWER:"""
        )
        
        # Function to format retrieved documents with metadata
        def format_docs(docs):
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata or {}
                source = metadata.get("filename", "Unknown source")
                timestamp = metadata.get("ingestion_timestamp", "")
                formatted_docs.append(
                    f"[Document {i}] Source: {source} {doc.page_content}"
                )
            return "\n\n".join(formatted_docs)

        # Function to clean the response by removing <think> tags
        def clean_response(response: str) -> str:
            """Remove <think> tags and their content from the response."""
            import re
            # Remove everything between <think> and </think> tags
            cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            # Remove any remaining <think> or </think> tags
            cleaned = re.sub(r'</?think>', '', cleaned)
            # Clean up extra whitespace
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            cleaned = cleaned.strip()
            return cleaned
        
        # Create an enhanced QA chain with document formatting and error handling
        qa_chain = (
            {
                "context": self.retriever | format_docs,  # Format documents with metadata
                "question": RunnablePassthrough()
            }
            | self.prompt 
            | chat_model 
            | StrOutputParser()
            | clean_response  # Clean the response to remove <think> tags
        )
        
        # Store the chain and retriever for later use
        self.qa_chain = qa_chain
    
    
    
    async def get_answer(self, question: str) -> dict:
        """Generate a complete answer for the given question using the QA chain."""
        logger.info(f"Generating answer for question: {question}")
        
        # Invoke the chain to get the answer
        start_time = time.time()
        
        try:
            # First retrieve the relevant documents to include in response
            docs = self.retriever.invoke(question)
            sources = []
            
            # Extract source information from documents
            for doc in docs:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Generate the answer using the QA chain
            answer = self.qa_chain.invoke(question)
            generation_time = time.time() - start_time
            logger.info(f"Generated answer in {generation_time:.4f}s")
            logger.debug(f"Raw answer before cleaning: {answer}")
            
            # Return both the answer and the sources
            return {
                "answer": answer,
                "sources": sources,
                "generation_time": generation_time
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "error": str(e)
            }


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
        vector_store.add_documents(split_docs)
        
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
async def query(q: str) -> JSONResponse:
    """
    Answer questions using the RAG QA chain.
    
    Args:
        q: Question string
        
    Returns:
        JSONResponse with the complete answer from the QA chain and source information
    
    Raises:
        HTTPException: If query is empty
    """
    # Validate query
    if not q.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Processing query: {q}")
    
    try:
        # Initialize retrieval service if not already done
        retrieval_service = RetrievalService()
        
        # Get answer from QA chain
        start_time = time.time()
        result = await retrieval_service.get_answer(q)
        process_time = time.time() - start_time
        
        # Return enhanced JSON response with sources
        return JSONResponse({
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "timestamp": datetime.now().isoformat(),
            "query": q,
            "process_time_seconds": round(process_time, 4)
        })
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
