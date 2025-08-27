import pymupdf as fitz
import asyncio
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List, Dict
import logging
from dotenv import load_dotenv
load_dotenv()

# Custom CLIP Embeddings class for FAISS
class CLIPEmbeddings(Embeddings):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.processor(
                text=text, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77
            )
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.squeeze().numpy().tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class MultimodalRAG:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_embeddings = CLIPEmbeddings()
        self.all_docs = []
        self.all_embeddings = []
        self.image_data_store = {}
        self.vector_store = None
        self.llm = None
        
    async def initialize_llm(self):
        """Initialize the Gemini language model"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"), 
                temperature=0.1
            )
            logging.info("Gemini LLM initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Gemini LLM: {e}")
            raise e

    def embed_image(self, image_data):
        """Embed image using CLIP"""
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else: 
            image = image_data
        
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()
        
    def embed_text(self, text):
        """Embed text using CLIP."""
        inputs = self.clip_processor(
            text=text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=77
        )
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()

    async def load_and_process_document(self, file_path: str):
        """Load and process document from file path"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logging.info(f"Processing document: {file_path}")
        self.all_docs = []
        self.all_embeddings = []
        self.image_data_store = {}
        doc = fitz.open(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        try:
            await self._process_pdf_pages(doc, splitter)
            await self._create_vector_store()
            doc.close()
            logging.info(f"Document processed successfully. Total docs: {len(self.all_docs)}")
        except Exception as e:
            doc.close()
            raise e
    
    async def _process_pdf_pages(self, doc, splitter):
        """Process all pages of the PDF"""
        for i, page in enumerate(doc):
            logging.info(f"Processing page {i+1}/{len(doc)}")
            
            ### process Text
            await self._process_page_text(page, i, splitter)
            ### Process image
            await self._process_page_images(doc, page, i)
    
    async def _process_page_text(self, page, page_num, splitter):
        """Process text content from a page"""
        text = page.get_text()
        if text.strip():
            temp_doc = Document(
                page_content=text, 
                metadata={"page": page_num, "type": "text"}
            )
            text_chunks = splitter.split_documents([temp_doc])
            
            for chunk in text_chunks:
                embedding = self.embed_text(chunk.page_content)
                self.all_embeddings.append(embedding)
                self.all_docs.append(chunk)
    
    async def _process_page_images(self, doc, page, page_num):
        """Process images from a page"""
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{page_num}_img_{img_index}"
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                self.image_data_store[image_id] = img_base64
                
                # Embed image using CLIP
                embedding = self.embed_image(pil_image)
                self.all_embeddings.append(embedding)
                
                # Create document for image
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": page_num, "type": "image", "image_id": image_id}
                )
                self.all_docs.append(image_doc)
                
            except Exception as e:
                logging.error(f"Error processing image {img_index} on page {page_num}: {e}")
                continue
    
    async def _create_vector_store(self):
        """Create FAISS vector store from processed documents"""
        if not self.all_docs:
            raise ValueError("No documents to create vector store")
        
        ### creating faiss vector store
        texts = [doc.page_content for doc in self.all_docs]
        metadatas = [doc.metadata for doc in self.all_docs]
        
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.clip_embeddings,
            metadatas=metadatas
        )
        logging.info("Vector store created successfully")
    
    async def retrieve_multimodal(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents using CLIP embeddings"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process a document first.")
        
        ### using Faiss for store and similarity search
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def create_multimodal_content(self, query: str, retrieved_docs: List[Document]) -> List[Dict]:
        """Create multimodal content for Gemini"""
        content = []
        
        # Add the query and context introduction
        context_text = f"Question: {query}\n\nContext:\n"
        text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
        image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
        if text_docs:
            text_context = "\n\n".join([
                f"[Page {doc.metadata['page']}]: {doc.page_content}"
                for doc in text_docs
            ])
            context_text += f"Text excerpts:\n{text_context}\n"
        
        ### adding image description
        if image_docs:
            context_text += f"\nImages from document (see attached images):\n"
            for doc in image_docs:
                context_text += f"- Image from page {doc.metadata['page']}\n"
        
        context_text += "\n\nPlease answer the question based on the provided text and images."
        content.append({
            "type": "text", 
            "text": context_text
        })
        for doc in image_docs:
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in self.image_data_store:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.image_data_store[image_id]}"
                    }
                })
        
        return content
    
    async def query_document(self, query: str, k: int = 5) -> str:
        """Main pipeline for multimodal RAG query"""
        if not self.llm:
            await self.initialize_llm()
        context_docs = await self.retrieve_multimodal(query, k=k)
        content = self.create_multimodal_content(query, context_docs)
        message = HumanMessage(content=content)
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, [message])
            logging.info(f"Retrieved {len(context_docs)} documents:")
            for doc in context_docs:
                doc_type = doc.metadata.get("type", "unknown")
                page = doc.metadata.get("page", "?")
                if doc_type == "text":
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    logging.info(f"  - Text from page {page}: {preview}")
                else:
                    logging.info(f"  - Image from page {page}")
            
            return response.content
            
        except Exception as e:
            logging.error(f"Error getting response: {e}")
            return await self._fallback_text_only_query(query, context_docs)

    async def _fallback_text_only_query(self, query: str, context_docs: List[Document]) -> str:
        text_docs = [doc for doc in context_docs if doc.metadata.get("type") == "text"]
        
        if not text_docs:
            return "No text content found to answer the question."
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        
        prompt = f"""Question: {query}

Context:
{text_context}

Please answer the question based on the provided text context."""

        message = HumanMessage(content=prompt)
        response = await asyncio.to_thread(self.llm.invoke, [message])
        return response.content

    async def process_user_document(self, file_path: str) -> bool:
        """Process document uploaded by user"""
        try:
            await self.load_and_process_document(file_path)
            return True
        except Exception as e:
            logging.error(f"Error processing user document: {e}")
            return False

rag = MultimodalRAG()
async def main():
    rag = MultimodalRAG()
    file_path = input("Enter the path to your PDF document: ").strip()
    print("Processing document...")
    success = await rag.process_user_document(file_path)
    
    if not success:
        print("Failed to process document. Please check the file path and try again.")
        return
    
    print("Document processed successfully!")
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        try:
            print("Searching for relevant information...")
            answer = await rag.query_document(query)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error processing query: {e}")
            logging.error(f"Query error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("==========================================")
    asyncio.run(main())