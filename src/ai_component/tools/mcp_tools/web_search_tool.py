import asyncio
import os
import sys
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.ai_component.config import Config
from src.ai_component.exception import CustomException
from src.ai_component.logger import logging

mcp = FastMCP("WebSearch")

@mcp.tool()
async def web_search(query: str) -> str:
    """Tool for web search on the given topics"""
    if not Config.serper_api_key:
        logging.warning("Serper Api Key not configured")
        return "Error: serper api key not configured correctly"
    try:
        logging.info(f'Performing Serper API web search for: {query}')
        
        headers = {
            'X-API-KEY': Config.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': 5  
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://google.serper.dev/search", 
                json=payload, 
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
        formatted_results = []
        
        # Check for answer box (direct answer)
        if "answerBox" in result:
            answer_box = result["answerBox"]
            if "answer" in answer_box:
                formatted_results.append(f"Direct Answer: {answer_box['answer']}")
            elif "snippet" in answer_box:
                formatted_results.append(f"Featured Snippet: {answer_box['snippet']}")
        
        # Get organic results
        if "organic" in result and result["organic"]:
            formatted_results.append("\nTop Search Results:")
            for i, item in enumerate(result["organic"][:3], 1):  # Top 3 results
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No snippet available")
                link = item.get("link", "No link")
                formatted_results.append(f"{i}. {title}\n   {snippet}\n   Source: {link}\n")
        
        if formatted_results:
            return "\n".join(formatted_results)
        else:
            return "No results found for the given query."
        
    except Exception as e:
        logging.error(f"Error in MCP web search: {e}")
        raise CustomException(e, sys) from e