import streamlit as st
import requests
import time
import uuid
from typing import Optional
import json

st.set_page_config(
    page_title="Multi-Modal Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

FASTAPI_URL = "http://localhost:8000" 

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.status-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.status-success {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
}
.status-error {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
}
.status-warning {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
}
.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0.5rem;
}
.user-message {
    background-color: #e3f2fd;
    margin-left: 2rem;
}
.assistant-message {
    background-color: #f5f5f5;
    margin-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def upload_document(file):
    """Upload document to FastAPI backend"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{FASTAPI_URL}/upload", files=files, timeout=60)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def query_document(query: str, session_id: Optional[str] = None):
    """Send query to FastAPI backend"""
    try:
        payload = {"query": query}
        if session_id:
            payload["session_id"] = session_id
            
        response = requests.post(
            f"{FASTAPI_URL}/query",
            json=payload,
            timeout=120
        )
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_system_status():
    """Get system status from FastAPI backend"""
    try:
        response = requests.get(f"{FASTAPI_URL}/status", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def reset_system():
    """Reset the system"""
    try:
        response = requests.delete(f"{FASTAPI_URL}/reset", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

# Main app
def main():
    st.markdown('<h1 class="main-header">🔍 Multi-Modal Research Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        api_healthy, health_data = check_api_health()
        
        if api_healthy:
            st.markdown('<div class="status-box status-success">✅ API Connected</div>', unsafe_allow_html=True)
            
            # Get detailed status
            status_ok, status_data = get_system_status()
            if status_ok:
                workflow_ready = status_data.get("workflow_ready", False)
                document_uploaded = status_data.get("document_uploaded", False)
                rag_initialized = status_data.get("rag_initialized", False)
                
                st.write(f"**Workflow:** {'✅' if workflow_ready else '❌'}")
                st.write(f"**Document:** {'✅' if document_uploaded else '❌'}")
                st.write(f"**RAG System:** {'✅' if rag_initialized else '❌'}")
                
                st.session_state.document_uploaded = document_uploaded
        else:
            st.markdown('<div class="status-box status-error">❌ API Disconnected</div>', unsafe_allow_html=True)
            st.error("Please make sure the FastAPI backend is running on http://localhost:8000")
            st.stop()
        
        st.divider()
        
        # Document Upload Section
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=['pdf'],
            help="Upload a PDF document to analyze and query"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing document... This may take a few minutes."):
                    success, result = upload_document(uploaded_file)
                    
                if success:
                    st.success("Document uploaded successfully! Processing in background...")
                    time.sleep(2)  # Give some time for processing to start
                    st.rerun()
                else:
                    st.error(f"Failed to upload document: {result.get('error', 'Unknown error')}")
        
        # Reset System
        st.divider()
        if st.button("🔄 Reset System", type="secondary"):
            with st.spinner("Resetting system..."):
                success, result = reset_system()
            
            if success:
                st.success("System reset successfully!")
                st.session_state.messages = []
                st.session_state.document_uploaded = False
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Failed to reset system: {result.get('error', 'Unknown error')}")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        if not st.session_state.document_uploaded:
            st.warning("⚠️ Please upload and process a PDF document before querying.")
            st.info("👈 Use the sidebar to upload your document")
        else:
            st.success("✅ Document ready for querying!")
        
        # Display chat messages
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'''
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>
                        {message["content"]}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong><br>
                        {message["content"]}
                    </div>
                    ''', unsafe_allow_html=True)
        
        # Query input
        with st.form("query_form", clear_on_submit=True):
            query = st.text_area(
                "Ask a question about your document:",
                placeholder="e.g., What is this document about? Can you summarize the key findings?",
                height=100,
                disabled=not st.session_state.document_uploaded
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                submit_button = st.form_submit_button(
                    "🔍 Ask Question", 
                    type="primary",
                    disabled=not st.session_state.document_uploaded
                )
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat")
        
        # Handle form submission
        if submit_button and query.strip():
            st.session_state.messages.append({
                "role": "user",
                "content": query
            })
            
            # Show spinner and get response
            with st.spinner("Thinking... This may take a moment as I search through your document and the web."):
                success, result = query_document(query, st.session_state.session_id)
            
            if success and result.get("success"):
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["response"]
                })
            else:
                error_msg = result.get("error", "Unknown error occurred")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Sorry, I encountered an error: {error_msg}"
                })
            
            st.rerun()
        
        if clear_button:
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        # st.header("ℹ️ Instructions")
        
        # st.markdown("""
        # ### How to use:
        
        # 1. **Upload Document** 📄
        #    - Click "Browse files" in the sidebar
        #    - Select a PDF document
        #    - Click "Process Document"
        #    - Wait for processing to complete
        
        # 2. **Ask Questions** 💭
        #    - Type your question in the text area
        #    - Click "Ask Question"
        #    - Wait for the AI to search and respond
        
        # 3. **Features** ✨
        #    - **Multimodal RAG**: Analyzes text and images
        #    - **Web Search**: Gets current information
        #    - **arXiv Papers**: Finds relevant research
        #    - **Browser Control**: Can interact with websites
        
        # ### Sample Questions:
        # - "What is this document about?"
        # - "Summarize the key findings"
        # - "What are the main conclusions?"
        # - "Find recent research related to this topic"
        # - "What do the images/charts show?"
        
        # ### Tips:
        # - Be specific in your questions
        # - Ask follow-up questions for clarity
        # - Use "Reset System" to start fresh
        # """)
        
        # Session info
        # st.divider()
        # st.subheader("🔧 Session Info")
        # st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        # st.text(f"Messages: {len(st.session_state.messages)}")
        
        # API endpoints info
        with st.expander("🔗 API Endpoints"):
            st.code(f"""
            FastAPI Backend: {FASTAPI_URL}
            
            Endpoints:
            - GET  /health
            - POST /upload
            - POST /query
            - GET  /status
            - DELETE /reset
            """)

# Error handling for the entire app
try:
    main()
except Exception as e:
    st.error("An unexpected error occurred!")
    st.exception(e)
    
    with st.expander("🐛 Debug Info"):
        st.write("Session State:", st.session_state)
        st.write("Error:", str(e))

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Multi-Modal Research Assistant v1.0 | Built with Streamlit + FastAPI
</div>
""", unsafe_allow_html=True)