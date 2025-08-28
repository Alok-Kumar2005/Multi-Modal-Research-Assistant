import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import asyncio
import aiosqlite
from functools import lru_cache
from langchain_core.messages import HumanMessage
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

async def test_workflow():
    """Test the complete workflow"""
    print("\nTesting Complete Workflow...")
    
    try:
        workflow = await create_workflow()
        test_query = "Tell me about artificial intelligence"
        print(f"Testing query: {test_query}")
        config = {"configurable": {"thread_id": "test_thread"}}
        
        result = await workflow.ainvoke(
            {"messages": [HumanMessage(content=test_query)]},
            config=config
        )
        
        print("Workflow completed successfully!")
        print(f"Final response: {result['messages'][-1].content}")
        
        return True
        
    except Exception as e:
        print(f"Error testing workflow: {e}")
        return False
    
async def main():
    workflow_success = await test_workflow()
    if not workflow_success:
        print("Workflow test failed. Please check logs.")
        return

if __name__ == "__main__":
    asyncio.run(main())