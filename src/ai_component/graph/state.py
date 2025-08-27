from typing import TypedDict,Annotated, Optional
from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AssistantState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    vector_response: Optional[str] = None
    research_response: Optional[str] = None