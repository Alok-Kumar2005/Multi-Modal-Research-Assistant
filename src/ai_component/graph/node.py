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
from src.ai_component.tools.vector_seach_tool import rag_tool
from src.ai_component.modules.RAG.vector_store import rag
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

class Nodes:
    @staticmethod
    async def QueryRefinerNode(state: AssistantState) -> dict:
        """
        Refine the user query
        """
        logging.info("Query refiner node ...................")
        try:
            query = state['messages'][-1].content if state['messages'] else ""
            if not query:
                logging.warning("No query found in state messages")
                return {
                    "messages": [AIMessage(content="Error: No query provided to refine")]
                }
                
            prompt = PromptTemplate(
                input_variables=['user_query'],
                template=Prompts.query_refiner_template
            )
            factory = LLMChainFactory(model_type="groq")
            chain = await factory.get_llm_chain_async(prompt=prompt)
            response = await chain.ainvoke({
                "user_query": query
            })
            return {
                "messages": [AIMessage(content=response.content)]
            }
        except Exception as e:
            logging.error(f"Error in Query Refiner Node: {e}")
            raise CustomException(e, sys) from e
        
    @staticmethod
    async def ResearchNode(state: AssistantState) -> dict:
        """Research on the Web, arxiv and take control of chrome"""
        logging.info("Research Node ..............")
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            
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
                        "args": [os.path.join(base_dir, "src", "ai_component", "tools", "mcp_tools", "web_search_tool.py")],
                        "transport": "stdio"
                    },
                    "browser_agent": {
                        "command": "python",
                        "args": [os.path.join(base_dir, "src", "ai_component", "tools", "mcp_tools", "browser_use_tool.py")],
                        "transport": "stdio"
                    }
                }
            )

            tools = await client.get_tools()
            factory = LLMChainFactory()
            llm = await factory.get_llm_async()
            
            agent = create_react_agent(
                llm, 
                tools=tools,
                system_prompt=Prompts.research_template
            )
            
            refined_query = state['messages'][-1].content if state['messages'] else ""
            if not refined_query:
                logging.warning("No refined query found in state messages")
                return {
                    "research_response": "Error: No query provided for research"
                }
            
            response = await agent.ainvoke({
                "messages": [HumanMessage(content=refined_query)]
            })
            
            # Extract the content from the response properly
            response_content = ""
            if hasattr(response, 'content'):
                response_content = response.content
            elif 'messages' in response and response['messages']:
                last_message = response['messages'][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)
            else:
                response_content = str(response)
            
            return {
                "research_response": response_content
            }
            
        except Exception as e:
            logging.error(f"Error in Research Agent: {e}")
            raise CustomException(e, sys) from e
        

    async def VectorNode(state: AssistantState)->dict:
        """Get the data from the vector store"""
        logging.info("Vector Node .................")
        try:
            query = state['messages'][-1].content if state['messages'] else ""
            response = await rag.query_document(query=query)

            return {
                "vector_response": response.content
            }
        except CustomException as e:
            logging.error(f"Error in Vector Node: {e}")
            raise CustomException(e, sys) from e
    
    ## node with tool call
    @staticmethod
    async def VectorNode2(state: AssistantState)->dict:
        """Get the relavent data from the vector database"""
        logging.info("Vector Node...............")
        try:
            query = state['messages'][-1].content if state['messages'] else ""
            prompt = PromptTemplate(
                input_variables= = ['user_query'],
                template= Prompts.vector_search_template
            )
            factory = LLMChainFactory(model_type= "gemini")
            chain = await factory.get_llm_tool_chain(
                prompt= prompt,
                tools= [rag_tool]
            )
            response = await chain.ainvoke({
                "user_query": query
            })

            return {
                "vector_response": response.content
            }
        except CustomException as e:
            logging.error(f"Error in Vector Node: {e}")
            raise CustomException(e, sys) from e
        

    