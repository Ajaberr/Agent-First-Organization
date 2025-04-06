import logging
import os
from typing import Dict, Any

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.graphs import StateGraph
from langchain.chains.graph import StateflowChain
from langchain.nodes import node_from_function

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.utils import chunk_string
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP

logger = logging.getLogger(__name__)

citation_prompt = """You are a citation generator specialized in climate research and datasets. 
You will help format citations for climate research papers, datasets, and other resources in the requested style.

User's input: {message}

Previous conversation: {history}

Given the user's input, please generate a properly formatted citation.
If the user doesn't specify a citation style, use APA 7th edition.
If the user doesn't provide enough information, ask specific questions to gather the necessary details.

Include all available information: author names, publication year, title, journal/source, volume, issue, page numbers, DOI, and URL as appropriate.

You must provide a properly formatted citation. Do not explain how to create citations or provide general information about citation styles unless explicitly asked.
"""

@register_worker
class CitationGeneratorWorker(BaseWorker):
    description = "Generates properly formatted citations for climate research papers, datasets, and other scientific resources in various citation styles (APA, MLA, Chicago, etc.)"

    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self._build_action_graph()

    def _generate_citation(self, state: MessageState) -> Dict[str, Any]:
        """Generate a citation based on user input"""
        user_message = state["user_message"]
        
        prompt = PromptTemplate.from_template(citation_prompt)
        input_prompt = prompt.invoke({
            "message": user_message.message, 
            "history": user_message.history
        })
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        
        final_chain = self.llm | StrOutputParser()
        citation = final_chain.invoke(chunked_prompt)
        
        # Update message state
        state["status"] = StatusEnum.completed
        state["answer"] = citation
        return state

    def _build_action_graph(self):
        """Build the action graph for citation generation"""
        generate_citation_node = node_from_function(self._generate_citation, name="generate_citation")
        
        # Create the workflow
        workflow = StateGraph(MessageState)
        workflow.add_node("generate_citation", generate_citation_node)
        
        # Add entry point
        workflow.set_entry_point("generate_citation")
        
        self.action_graph = workflow

    def execute(self, msg_state: MessageState):
        """Execute the citation generator workflow"""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result 