import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from langchain.tools import BaseTool
from typing import Type, ClassVar
import asyncio
import hashlib
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from src.ai_component.modules.RAG.vector_store import rag
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException


class ToolInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant information in the RAG system.")


class RAGTool(BaseTool):
    name: str = "rag_tool"
    description: str = "Tool to search in the relavent documents"
    args_schema: Type[ToolInput] = ToolInput

    async def _arun(self, query: str) ->str:
        """Async way to retrieve the similar document"""
        try:
            logging.info("Calling RAG Tool...........")
            answer = await rag.query_document(query)

            return answer
        except CustomException as e:
            logging.info(f"Error in RAG Tool: {e}")
            raise CustomException(e, sys) from e
        
    def _run(self, query: str)->str:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun(query))
        except CustomException as e:
            logging.error(f"Error in RAG tool: {e}")
            raise CustomException(e, sys) from e
        

rag_tool = RAGTool()