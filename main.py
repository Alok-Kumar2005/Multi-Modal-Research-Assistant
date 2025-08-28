import uvicorn
import os
import sys
import asyncio
import uuid
from typing import Optional
import aiofiles
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai_component.graph.graph import create_workflow
from src.ai_component.modules.RAG.vector_store import rag
from src.ai_component.logger import logging
from langchain_core.messages import HumanMessage

load_dotenv()

workflow = None
document_uploaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    global workflow
    logging.info("Starting up FastAPI application...")
    try:
        workflow = await create_workflow()
        logging.info("Workflow created successfully")
    except Exception as e:
        logging.error(f"Failed to create workflow: {e}")
        raise
    
    yield
    logging.info("Shutting down FastAPI application...")

app = FastAPI(
    title="Multi-Modal Research Assistant API",
    description="API for document upload, processing, and intelligent querying",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    success: bool
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    filename: str
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    document_uploaded: bool
    workflow_ready: bool

def ensure_upload_dir():
    """Ensure upload directory exists"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

async def process_document_background(file_path: str, filename: str):
    """Background task to process uploaded document"""
    global document_uploaded
    
    try:
        logging.info(f"Processing document: {filename}")
        success = await rag.process_user_document(file_path)
        
        if success:
            document_uploaded = True
            logging.info(f"Document {filename} processed successfully")
        else:
            logging.error(f"Failed to process document: {filename}")
            
    except Exception as e:
        logging.error(f"Error processing document {filename}: {e}")
        document_uploaded = False
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="running",
        document_uploaded=document_uploaded,
        workflow_ready=workflow is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if workflow is not None else "unhealthy",
        document_uploaded=document_uploaded,
        workflow_ready=workflow is not None
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a PDF document"""
    global document_uploaded
    document_uploaded = False
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    if file.size and file.size > 50 * 1024 * 1024: 
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum 50MB allowed."
        )
    
    try:
        upload_dir = ensure_upload_dir()
        file_path = os.path.join(upload_dir, file.filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        background_tasks.add_task(
            process_document_background,
            file_path,
            file.filename
        )
        
        return UploadResponse(
            message=f"Document {file.filename} uploaded successfully. Processing in background...",
            filename=file.filename,
            success=True
        )
        
    except Exception as e:
        logging.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query the processed document"""
    global workflow, document_uploaded
    
    if not workflow:
        raise HTTPException(
            status_code=500,
            detail="Workflow not initialized"
        )
    
    if not document_uploaded:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded and processed yet. Please upload a document first."
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        result = await workflow.ainvoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config
        )
        if result and 'messages' in result and result['messages']:
            response_content = result['messages'][-1].content
        else:
            response_content = "No response generated"
        
        return QueryResponse(
            response=response_content,
            session_id=session_id,
            success=True
        )
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return QueryResponse(
            response="",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error=str(e)
        )

@app.get("/status")
async def get_status():
    """Get current system status"""
    return {
        "workflow_ready": workflow is not None,
        "document_uploaded": document_uploaded,
        "rag_initialized": hasattr(rag, 'vector_store') and rag.vector_store is not None
    }

@app.delete("/reset")
async def reset_system():
    """Reset the system (clear uploaded document)"""
    global document_uploaded
    
    try:
        document_uploaded = False
        rag.vector_store = None
        rag.all_docs = []
        rag.all_embeddings = []
        rag.image_data_store = {}
        
        return {"message": "System reset successfully"}
        
    except Exception as e:
        logging.error(f"Error resetting system: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset system: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )