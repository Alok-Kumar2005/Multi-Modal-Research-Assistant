import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

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
                template= Prompts.query_refiner_prompt
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
        
    