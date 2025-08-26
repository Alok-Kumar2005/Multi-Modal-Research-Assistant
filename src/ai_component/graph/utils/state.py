from typing import TypedDict,Annotated, Optional, Literal, List, Dict, Any
from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AssistantState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    uploaded_files: Optional[List[str]]
    session_id: str
    
    # Input Classification
    input_type: Optional[Literal["text", "image", "voice", "document", "mixed"]]
    processing_needed: Optional[List[str]]  # ["image", "document", "web_search"]
    
    # Processing Results
    processed_image: Optional[Dict[str, Any]]
    processed_voice: Optional[str]  # Transcribed text
    processed_documents: Optional[List[Document]]
    extracted_text: Optional[str]  # OCR or voice transcription
    
    # Query Analysis
    query_intent: Optional[str]  # "research", "compare", "analyze", "summarize"
    entities_detected: Optional[List[str]]  # Company names, products, etc.
    requires_current_data: Optional[bool]
    requires_user_docs: Optional[bool]
