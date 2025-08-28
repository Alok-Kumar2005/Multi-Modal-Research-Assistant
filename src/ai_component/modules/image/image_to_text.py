import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import base64
import tempfile
from pathlib import Path
from typing import Union, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.ai_component.config import Config
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException


class ImageToTextProcessor:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        try:
            self.model_name = model_name
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=Config.gemini_api_key
            )
            logging.info(f"ImageToTextProcessor initialized with model: {model_name}")
        except Exception as e:
            logging.error(f"Error initializing ImageToTextProcessor: {str(e)}")
            raise CustomException(e, sys) from e

    def _encode_image_bytes(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def _encode_image_from_path(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return self._encode_image_bytes(image_file.read())
        except FileNotFoundError:
            raise CustomException(f"Image file not found: {image_path}", sys)
        except Exception as e:
            raise CustomException(f"Error reading image file: {str(e)}", sys)

    def _get_image_mime_type(self, image_path: Optional[str] = None, 
                           image_bytes: Optional[bytes] = None) -> str:
        if image_path:
            suffix = Path(image_path).suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            return mime_types.get(suffix, 'image/jpeg')
        return 'image/jpeg'  # Default fallback

    def process_image_url(self, image_url: str, prompt: str = "Describe this image in detail.") -> Dict[str, Any]:
        try:
            if not image_url or not image_url.strip():
                return {
                    "success": False,
                    "error": "No image URL provided"
                }

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url}
                ]
            )

            logging.info(f"Processing image from URL with prompt: {prompt[:50]}...")
            result = self.llm.invoke([message])
            
            return {
                "success": True,
                "text": result.content,
                "image_source": "url",
                "prompt_used": prompt
            }

        except Exception as e:
            logging.error(f"Error processing image from URL: {str(e)}")
            return {
                "success": False,
                "error": f"Image processing error: {str(e)}"
            }

    def process_image_bytes(self, image_bytes: bytes, 
                          prompt: str = "Describe this image in detail.",
                          image_format: str = "jpeg") -> Dict[str, Any]:
        try:
            if not image_bytes or len(image_bytes) == 0:
                return {
                    "success": False,
                    "error": "No image data provided"
                }
            if len(image_bytes) < 1000:
                return {
                    "success": False,
                    "error": "Image file is too small"
                }
            encoded_image = self._encode_image_bytes(image_bytes)
            data_url = f"data:image/{image_format};base64,{encoded_image}"

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": data_url}
                ]
            )

            logging.info(f"Processing image from bytes with prompt: {prompt[:50]}...")
            result = self.llm.invoke([message])
            
            return {
                "success": True,
                "text": result.content,
                "image_source": "bytes",
                "prompt_used": prompt,
                "image_size_bytes": len(image_bytes)
            }

        except Exception as e:
            logging.error(f"Error processing image from bytes: {str(e)}")
            return {
                "success": False,
                "error": f"Image processing error: {str(e)}"
            }

    def process_image_file(self, image_path: str, 
                         prompt: str = "Describe this image in detail.") -> Dict[str, Any]:
        try:
            if not image_path or not image_path.strip():
                return {
                    "success": False,
                    "error": "No image path provided"
                }

            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size < 1000:  # Less than 1KB
                return {
                    "success": False,
                    "error": "Image file is too small"
                }

            ### encoding image
            encoded_image = self._encode_image_from_path(image_path)
            mime_type = self._get_image_mime_type(image_path=image_path)
            data_url = f"data:{mime_type};base64,{encoded_image}"

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": data_url}
                ]
            )

            logging.info(f"Processing image from file: {image_path} with prompt: {prompt[:50]}...")
            result = self.llm.invoke([message])
            
            return {
                "success": True,
                "text": result.content,
                "image_source": "file",
                "image_path": image_path,
                "prompt_used": prompt,
                "file_size_bytes": file_size
            }

        except Exception as e:
            logging.error(f"Error processing image from file: {str(e)}")
            return {
                "success": False,
                "error": f"Image processing error: {str(e)}"
            }

    def process_text_only(self, text: str) -> Dict[str, Any]:
        try:
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "No text provided"
                }

            message = HumanMessage(content=text)
            
            logging.info(f"Processing text-only input: {text[:50]}...")
            result = self.llm.invoke([message])
            
            return {
                "success": True,
                "text": result.content,
                "input_type": "text_only",
                "original_text": text
            }

        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return {
                "success": False,
                "error": f"Text processing error: {str(e)}"
            }

processor = ImageToTextProcessor()
if __name__ == "__main__":
    print("=== Testing Image URL Processing ===")
    result_url = processor.process_image_url(
        image_url="https://picsum.photos/seed/picsum/400/300",
        prompt="What do you see in this image? Describe it in detail."
    )
    if result_url["success"]:
        print("URL Result:", result_url["text"])
    else:
        print("URL Error:", result_url["error"])


    print("\n=== Testing Text-Only Processing ===")
    result_text = processor.process_text_only("Explain the concept of machine learning in simple terms.")
    if result_text["success"]:
        print("Text Result:", result_text["text"])
    else:
        print("Text Error:", result_text["error"])
    