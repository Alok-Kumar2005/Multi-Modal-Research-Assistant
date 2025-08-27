import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import asyncio
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from src.ai_component.graph.state import AssistantState
from src.ai_component.llm import LLMChainFactory
from src.ai_component.core.prompts import Prompts
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

class Nodes:
    @staticmethod
    async def QueryRefinerNode(state: AssistantState)->dict:
        """
        Refine the user query
        """
        logging.info("Query refiner node ...................")
        try:
            query = state['messages'][-1].content if state['messages'] else ""
            prompt = PromptTemplate(
                input_variables= ['user_query'],
                template= Prompts.query_refiner_template
            )
            factory = LLMChainFactory(model_type= "groq")
            chain = await factory.get_llm_chain_async(prompt = prompt)
            response = await chain.ainvoke({
                "user_query": query
            })
            return {
                "messages": [AIMessage(content = response.content)]
            }
        except CustomException as e:
            logging.error(f"Error in Query Refiner Node: {e}")
            raise CustomException(e, sys) from e
        
    @staticmethod
    async def ResearchNode(state: AssistantState)->dict:
        """Research on the Web,arxiv and take control of chrome """
        logging.info("Reseach Node ..............")
        try:
            client = MultiServerMCPClient(
                {
                    "arXivPaper": {
                        "command": "uv",
                        "args": [
                        "run",
                        "--with",
                        "arxiv-paper-mcp>=0.1.0",
                        "arxiv-mcp"
                        ]
                    },
                    "WebSearch": {
                        "command": "python",
                        "args": ["src/ai_component/tools/mcp_tools/web_search_tool.py"],
                        "transport": "stdio"
                    },
                    "browser_agent":{
                        "command": "python",
                        "args": ["src\ai_component\tools\mcp_tools\browser_use_tool.py"],
                        "transport": "stdio"
                    }
                }
            )

            tools = await client.get_tools()
            factory = LLMChainFactory()
            llm = await factory.get_llm_async()
            agent = create_react_agent(
                llm , tools= tools,
                prompt= Prompts.research_template
            )
            refined_query = state['messages'][-1].content if state['messages'] else ""
            response = await agent.ainvoke({
                "messages": [{"role": "user", "content": refined_query}]
            })
            return {
                "research_response": response.content
            }
        except CustomException as e:
            logging.error(f" Error in Research Agent: {e}")
            raise CustomException(e, sys) from e
        