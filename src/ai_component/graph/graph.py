import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import asyncio
import aiosqlite
from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.ai_component.graph.state import AssistantState
from src.ai_component.graph.node import Nodes
from typing import Optional, List

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data2', 'chat_history.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
async_saver = None

async def initialize_database():
    """Initialize the async SQLite database and saver"""
    global async_saver
    conn = await aiosqlite.connect(DB_PATH, check_same_thread=False)
    async_saver = AsyncSqliteSaver(conn)
    await async_saver.setup()
    
    return async_saver

async def create_workflow():
    global async_saver
    
    if async_saver is None:
        async_saver = await initialize_database()
    
    graph_builder = StateGraph(AssistantState)
    graph_builder.add_node("query_refiner", Nodes.QueryRefinerNode)
    graph_builder.add_node("research_node", Nodes.ResearchNode)
    graph_builder.add_node("vector_node", Nodes.VectorNode)
    graph_builder.add_node("combined_node", Nodes.CombinedNode)

    graph_builder.add_edge(START, "query_refiner")
    graph_builder.add_edge("query_refiner", "research_node")
    graph_builder.add_edge("query_refiner", "vector_node")
    graph_builder.add_edge("research_node", "combined_node")
    graph_builder.add_edge("vector_node", "combined_node")
    graph_builder.add_edge("combined_node", END)

    return graph_builder.compile(checkpointer= async_saver)