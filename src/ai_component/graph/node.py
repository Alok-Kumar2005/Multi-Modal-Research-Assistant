import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.graph.state import AssistantState

from src.ai_component.logger import logging
from src.ai_component.exception import CustomException


class Nodes:
    def RouteNode(state: AssistantState)->dict:
        """
        Route the workflow on the basis of the user query
        """
        logging.info("Calling Route Node ............")
        processing_needed = []
        input_types = []

        if state['uploaded_files']:
            for file_path in state['uploaded_files']:
                file_ext = file_path.lower().split('.')[-1]
                if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    processing_needed.append("image")
                    input_types.append("image")
                elif file_ext in ['pdf', 'docx', 'txt', 'md']:
                    processing_needed.append("document")
                    input_types.append("document")
                elif file_ext in ['mp3', 'wav', 'm4a', 'ogg']:
                    processing_needed.append("voice")
                    input_types.append("voice")

        processing_needed.append("query_analysis")
        if len(input_types) == 0:
            primary_type = "text"
        elif len(input_types) == 1:
            primary_type = input_types[0]
        else:
            primary_type = "mixed"

        return {
            "input_type"
        }