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
async def web_search(query: str)-> str:
    """Tool for web search on the given topics """
    if not Config.serper_api_key:
        logging.warning("Serper Api Key not configured")
        return "Error: serper api key not configured correctly"
    try:
        logging.info('Performing Serper api web search')
        params = {
            "q": query,
            "api_key": Config.serper_api_key,
            "source": "python"
        }
        async with httpx.AsyncClient() as client:
            response = await client.get("https://serpapi.com/search", params= params)
            response.raise_for_status()
            result = response.json()

        ## extracting top results
        if "answer_box" in result and "snippet" in result['answer_box']:
            return f"Result from Web Search : {result['answer_box']['snippet']}"
        elif "organic_results" in result and result['organic_results']:
            top_result = result['organic_results'][0]
            return f"Top search result : {top_result.get('snippet', 'No snippet available.')} (Source: {top_result.get('link')})"
        else:
            return "No direct result found for the topic"
        
    except CustomException as e:
        logging.error(f"Error in Mcp web search : {e}")
        raise CustomException(e, sys) from e
    

if __name__ == "__main__":
    logging.info("Web Searching Started")
    mcp.run(transport="stdio")