import streamlit as st
import requests
import time
import uuid
from typing import Optional
import json
import sys
import os
from audio_recorder_streamlit import audio_recorder
from PIL import Image
import io

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
try:
    from src.ai_component.modules.audio.speechTotext import AudioTranscriber
    from src.ai_component.modules.image.image_to_text import ImageToTextProcessor
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

st.set_page_config(
    page_title="Multi-Modal Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

FASTAPI_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1.5rem;
}

.status-box {
    padding: 0.8rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    font-weight: 500;
}

.status-success {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
    color: #155724;
}

.status-error {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
    color: #721c24;
}

.chat-container {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    height: 500px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.8rem 1rem;
    border-radius: 15px 15px 5px 15px;
    margin: 0.5rem 0 0.5rem 2rem;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.assistant-message {
    background: white;
    color: #333;
    padding: 0.8rem 1rem;
    border-radius: 15px 15px 15px 5px;
    margin: 0.5rem 2rem 0.5rem 0;
    max-width: 80%;
    border: 1px solid #e1e5e9;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

.input-container {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid #e1e5e9;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.audio-button {
    background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    color: white;
    font-size: 1.2rem;
    cursor: pointer;
    transition: all 0.3s;
}

.audio-button:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Initialize processors
@st.cache_resource
def get_processors():
    try:
        transcriber = AudioTranscriber()
        image_processor = ImageToTextProcessor()
        return transcriber, image_processor
    except Exception as e:
        st.error(f"Failed to initialize processors: {e}")
        return None, None

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

def main():
    st.markdown('<h1 class="main-header">🔍 Multi-Modal Research Assistant</h1>', unsafe_allow_html=True)
    
    # Get processors
    transcriber, image_processor = get_processors()
    if transcriber is None or image_processor is None:
        st.error("Failed to initialize audio/image processors. Please check your configuration.")
        st.stop()
    
    # Sidebar for document upload and status
    with st.sidebar:
        st.header("📊 System Status")
        
        # API Health Check
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
            help="Upload a PDF document for RAG processing"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing document... This may take a few minutes."):
                    success, result = upload_document(uploaded_file)
                    
                if success:
                    st.success("Document uploaded successfully! Processing in background...")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"Failed to upload document: {result.get('error', 'Unknown error')}")
        
        # System Controls
        st.divider()
        st.header("🔧 Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                with st.spinner("Resetting..."):
                    success, result = reset_system()
                
                if success:
                    st.success("System reset!")
                    st.session_state.messages = []
                    st.session_state.document_uploaded = False
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Reset failed: {result.get('error', 'Unknown error')}")
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    st.header("💬 Multi-Modal Chat")
    
    if not st.session_state.document_uploaded:
        st.info("👈 Upload a PDF document in the sidebar to get started with RAG capabilities!")
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                # Display user input type indicator
                input_type = message.get("input_type", "text")
                type_indicator = {
                    "text": "💬",
                    "audio": "🎤", 
                    "image": "🖼️",
                    "image_text": "🖼️💬"
                }.get(input_type, "💬")
                
                st.markdown(f'''
                <div class="user-message">
                    <small>{type_indicator} You:</small><br>
                    {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="assistant-message">
                    <small>🤖 Assistant:</small><br>
                    {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input interface
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Input tabs
    tab1, tab2, tab3 = st.tabs(["💬 Text", "🎤 Audio", "🖼️ Image"])
    
    with tab1:
        # Text input
        with st.form("text_form", clear_on_submit=True):
            text_input = st.text_area(
                "Type your message:",
                placeholder="Ask anything about your document or any topic...",
                height=100,
                key="text_input"
            )
            
            if st.form_submit_button("Send", type="primary", use_container_width=True):
                if text_input.strip():
                    process_user_input(text_input.strip(), "text")
    
    with tab2:
        # Audio input
        st.write("🎤 **Record your voice:**")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#ff6b6b",
            neutral_color="#6b6b6b",
            icon_name="microphone",
            icon_size="2x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🔄 Convert to Text & Send", type="primary", use_container_width=True):
                with st.spinner("Converting speech to text..."):
                    try:
                        result = transcriber.transcriber_bytes(audio_bytes)
                        
                        if result["success"]:
                            transcribed_text = result["text"]
                            st.success(f"Transcribed: {transcribed_text}")
                            process_user_input(transcribed_text, "audio")
                        else:
                            st.error(f"Transcription failed: {result['error']}")
                    except Exception as e:
                        st.error(f"Audio processing error: {str(e)}")
    
    with tab3:
        # Image input
        st.write("🖼️ **Upload an image:**")
        
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            key="image_uploader"
        )
        
        if uploaded_image:
            # Display image preview
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True, width=300)
            
            # Optional text with image
            image_text = st.text_area(
                "Optional: Add text with your image",
                placeholder="Ask something specific about this image...",
                height=60,
                key="image_text_input"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 Send Image Only", use_container_width=True):
                    process_image_input(uploaded_image, None, "image")
            
            with col2:
                if st.button("📤 Send Image + Text", use_container_width=True, disabled=not image_text.strip()):
                    if image_text.strip():
                        process_image_input(uploaded_image, image_text.strip(), "image_text")
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_user_input(user_input: str, input_type: str):
    """Process user text input"""
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "input_type": input_type
    })
    
    # Get response from backend
    with st.spinner("🤔 Thinking..."):
        success, result = query_document(user_input, st.session_state.session_id)
    
    if success and result.get("success"):
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

def process_image_input(uploaded_image, text_prompt: Optional[str], input_type: str):
    """Process image input with optional text"""
    try:
        # Get processors
        _, image_processor = get_processors()
        
        # Prepare image data
        image_bytes = uploaded_image.getvalue()
        
        # Determine prompt
        if text_prompt:
            prompt = text_prompt
            display_content = f"[Image uploaded]\n{text_prompt}"
        else:
            prompt = "Describe this image in detail and analyze its content."
            display_content = "[Image uploaded] - Please analyze this image"
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": display_content,
            "input_type": input_type
        })
        
        # Process image
        with st.spinner("🔍 Analyzing image..."):
            # Get image format
            image_format = uploaded_image.name.split('.')[-1].lower()
            if image_format == 'jpg':
                image_format = 'jpeg'
                
            result = image_processor.process_image_bytes(
                image_bytes=image_bytes,
                prompt=prompt,
                image_format=image_format
            )
        
        if result["success"]:
            # Combine image analysis with document query if needed
            image_analysis = result["text"]
            
            if text_prompt:
                # If there's text with image, combine both for final query
                combined_query = f"Based on the image I uploaded (which shows: {image_analysis}) and my question: {text_prompt}"
            else:
                combined_query = f"I uploaded an image that shows: {image_analysis}. Please provide insights about this image."
            
            # Query the document system with the image analysis
            if st.session_state.document_uploaded:
                with st.spinner("🔍 Searching documents and web..."):
                    success, query_result = query_document(combined_query, st.session_state.session_id)
                
                if success and query_result.get("success"):
                    response = query_result["response"]
                else:
                    response = f"Image Analysis: {image_analysis}\n\nNote: Could not connect to document system for additional context."
            else:
                response = f"**Image Analysis:**\n{image_analysis}"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Sorry, I couldn't analyze the image: {result['error']}"
            })
            
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Error processing image: {str(e)}"
        })
    
    st.rerun()

if __name__ == "__main__":
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
<div style="text-align: center; color: #666; padding: 1rem; font-size: 0.9rem;">
    🔍 Multi-Modal Research Assistant v2.0 | Built with Streamlit + FastAPI<br>
    <small>Supports Text 💬 | Audio 🎤 | Images 🖼️</small>
</div>
""", unsafe_allow_html=True)