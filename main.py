import uvicorn
import os
import sys
import asyncio
import uuid
from typing import Optional
import aiofiles
from contextlib import asynccontextmanager
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global variables
workflow = None
document_uploaded = False
initialization_error = None

# Import with error handling
try:
    from src.ai_component.graph.graph import create_workflow
    from src.ai_component.modules.RAG.vector_store import rag
    from src.ai_component.logger import logging
    from langchain_core.messages import HumanMessage
    
    logging.info("Successfully imported required modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    global workflow, initialization_error
    
    logging.info("Starting up FastAPI application...")
    
    try:
        # Initialize workflow with timeout
        logging.info("Creating workflow...")
        workflow = await asyncio.wait_for(create_workflow(), timeout=60.0)  # 60 second timeout
        logging.info("Workflow created successfully")
        
    except asyncio.TimeoutError:
        error_msg = "Workflow creation timed out after 60 seconds"
        logging.error(error_msg)
        initialization_error = error_msg
        
    except Exception as e:
        error_msg = f"Failed to create workflow: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Full traceback: {traceback.format_exc()}")
        initialization_error = error_msg
    
    # Yield control to FastAPI
    yield
    
    # Cleanup
    logging.info("Shutting down FastAPI application...")
    try:
        if workflow:
            # Add any cleanup logic here if needed
            pass
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Multi-Modal Research Assistant API",
    description="API for document upload, processing, and intelligent querying with multi-modal support",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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
    initialization_error: Optional[str] = None

def ensure_upload_dir():
    """Ensure upload directory exists"""
    upload_dir = "uploads"
    try:
        os.makedirs(upload_dir, exist_ok=True)
        return upload_dir
    except Exception as e:
        logging.error(f"Failed to create upload directory: {e}")
        return None

async def process_document_background(file_path: str, filename: str):
    """Background task to process uploaded document"""
    global document_uploaded
    
    try:
        logging.info(f"Starting background processing for document: {filename}")
        
        # Add timeout for document processing
        success = await asyncio.wait_for(
            rag.process_user_document(file_path), 
            timeout=300.0  # 5 minute timeout
        )
        
        if success:
            document_uploaded = True
            logging.info(f"Document {filename} processed successfully")
        else:
            logging.error(f"Failed to process document: {filename}")
            document_uploaded = False
            
    except asyncio.TimeoutError:
        logging.error(f"Document processing timed out for: {filename}")
        document_uploaded = False
        
    except Exception as e:
        logging.error(f"Error processing document {filename}: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        document_uploaded = False
        
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Cleaned up uploaded file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to clean up file {file_path}: {e}")

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - basic health check"""
    return HealthResponse(
        status="running",
        document_uploaded=document_uploaded,
        workflow_ready=workflow is not None,
        initialization_error=initialization_error
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    status = "healthy"
    
    if initialization_error:
        status = "unhealthy"
    elif workflow is None:
        status = "initializing"
    
    return HealthResponse(
        status=status,
        document_uploaded=document_uploaded,
        workflow_ready=workflow is not None,
        initialization_error=initialization_error
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a PDF document"""
    global document_uploaded
    
    # Reset document status
    document_uploaded = False
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Check file size
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum 50MB allowed."
        )
    
    try:
        # Ensure upload directory exists
        upload_dir = ensure_upload_dir()
        if not upload_dir:
            raise HTTPException(
                status_code=500,
                detail="Failed to create upload directory"
            )
        
        # Create unique filename to avoid conflicts
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save uploaded file
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            logging.info(f"File saved: {file_path} (size: {len(content)} bytes)")
            
        except Exception as e:
            logging.error(f"Failed to save file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded file: {str(e)}"
            )
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            file_path,
            file.filename
        )
        
        return UploadResponse(
            message=f"Document '{file.filename}' uploaded successfully. Processing in background...",
            filename=file.filename,
            success=True
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logging.error(f"Unexpected error during upload: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during upload: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query the processed document"""
    global workflow, document_uploaded, initialization_error
    
    # Check if workflow is available
    if workflow is None:
        if initialization_error:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow initialization failed: {initialization_error}"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Workflow is still initializing. Please wait a moment and try again."
            )
    
    # Validate query
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        
        # Prepare query content
        query_content = request.query.strip()
        if not document_uploaded:
            query_content = f"Note: No document has been uploaded for RAG. Please answer based on general knowledge and web search: {query_content}"
        
        logging.info(f"Processing query for session {session_id}: {query_content[:100]}...")
        
        # Process query with timeout
        result = await asyncio.wait_for(
            workflow.ainvoke(
                {"messages": [HumanMessage(content=query_content)]},
                config=config
            ),
            timeout=120.0  # 2 minute timeout
        )
        
        # Extract response
        if result and 'messages' in result and result['messages']:
            response_content = result['messages'][-1].content
            if not response_content:
                response_content = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        else:
            response_content = "I apologize, but I couldn't generate a response. Please try again."
        
        logging.info(f"Query processed successfully for session {session_id}")
        
        return QueryResponse(
            response=response_content,
            session_id=session_id,
            success=True
        )
        
    except asyncio.TimeoutError:
        logging.error(f"Query timed out for session {request.session_id}")
        return QueryResponse(
            response="",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error="Query timed out. Please try a simpler question or try again later."
        )
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return QueryResponse(
            response="",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error=f"An error occurred while processing your query: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get current system status"""
    try:
        rag_initialized = (
            hasattr(rag, 'vector_store') and 
            rag.vector_store is not None
        )
    except Exception:
        rag_initialized = False
    
    return {
        "workflow_ready": workflow is not None,
        "document_uploaded": document_uploaded,
        "rag_initialized": rag_initialized,
        "initialization_error": initialization_error
    }

@app.delete("/reset")
async def reset_system():
    """Reset the system (clear uploaded document and chat history)"""
    global document_uploaded
    
    try:
        logging.info("Resetting system...")
        document_uploaded = False
        
        # Reset RAG system
        try:
            if hasattr(rag, 'vector_store'):
                rag.vector_store = None
            if hasattr(rag, 'all_docs'):
                rag.all_docs = []
            if hasattr(rag, 'all_embeddings'):
                rag.all_embeddings = []
            if hasattr(rag, 'image_data_store'):
                rag.image_data_store = {}
                
            logging.info("RAG system reset successfully")
        except Exception as e:
            logging.error(f"Error resetting RAG system: {e}")
        
        # Clean up upload directory
        upload_dir = "uploads"
        if os.path.exists(upload_dir):
            try:
                for file in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logging.info("Upload directory cleaned")
            except Exception as e:
                logging.error(f"Error cleaning upload directory: {e}")
        
        logging.info("System reset completed")
        return {"message": "System reset successfully", "success": True}
        
    except Exception as e:
        logging.error(f"Error resetting system: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset system: {str(e)}"
        )

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logging.error(f"Unhandled exception: {exc}")
    logging.error(f"Full traceback: {traceback.format_exc()}")
    
    return {
        "error": "An unexpected error occurred",
        "detail": str(exc) if app.debug else "Please contact support"
    }

if __name__ == "__main__":
    # Run with better configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )