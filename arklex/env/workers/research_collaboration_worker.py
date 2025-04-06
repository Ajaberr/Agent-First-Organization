import logging
import os
from typing import Dict, Any, List

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.graphs import StateGraph
from langchain.nodes import node_from_function

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.utils import chunk_string
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP

logger = logging.getLogger(__name__)

# Mock database of climate research institutions and experts - in a real implementation, this would be fetched from an actual database
RESEARCH_INSTITUTIONS = [
    {
        "name": "Columbia Climate School",
        "location": "New York, NY, USA",
        "expertise": ["climate modeling", "climate policy", "climate impacts", "sustainable development"],
        "collaboration_opportunities": ["joint research projects", "student exchanges", "visiting scholar programs"],
        "contact": "climateschool@columbia.edu"
    },
    {
        "name": "Potsdam Institute for Climate Impact Research",
        "location": "Potsdam, Germany",
        "expertise": ["climate physics", "earth system analysis", "climate impacts", "adaptation strategies"],
        "collaboration_opportunities": ["international research partnerships", "PhD programs", "scientific workshops"],
        "contact": "cooperation@pik-potsdam.de"
    },
    {
        "name": "CSIRO Climate Science Centre",
        "location": "Aspendale, Australia",
        "expertise": ["ocean modeling", "atmospheric research", "climate extremes", "regional climate"],
        "collaboration_opportunities": ["research partnerships", "data sharing", "postdoctoral positions"],
        "contact": "climate.partnerships@csiro.au"
    },
    {
        "name": "National Center for Atmospheric Research (NCAR)",
        "location": "Boulder, CO, USA",
        "expertise": ["atmospheric science", "climate modeling", "earth system observations", "computational science"],
        "collaboration_opportunities": ["visitor program", "collaborative projects", "model development"],
        "contact": "collaborations@ucar.edu"
    },
    {
        "name": "Met Office Hadley Centre",
        "location": "Exeter, UK",
        "expertise": ["climate prediction", "earth system modeling", "climate monitoring", "climate services"],
        "collaboration_opportunities": ["international partnerships", "knowledge exchange", "joint appointments"],
        "contact": "research.partnerships@metoffice.gov.uk"
    }
]

research_collaboration_prompt = """You are a climate research collaboration specialist helping researchers connect with potential collaborators and institutions.

User's input: {message}
Previous conversation: {history}

Based on the user's research interests and needs, suggest appropriate collaboration opportunities from the following research institutions:

{institutions}

If needed, ask follow-up questions to better understand:
1. The user's specific research focus
2. What type of collaboration they're looking for (joint research, data sharing, etc.)
3. Geographic preferences or constraints
4. Timeline and scope of their project

For your recommendations, provide:
1. Institution name and location
2. Why this institution would be a good match based on expertise
3. Available collaboration opportunities
4. Initial contact information
5. Suggestions for how to approach the collaboration

Focus on making specific, tailored recommendations that align with the user's research goals in climate science.
"""

@register_worker
class ResearchCollaborationWorker(BaseWorker):
    description = "Helps climate researchers find suitable collaboration partners, research institutions, and funding opportunities for joint climate research projects"

    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self.institutions = RESEARCH_INSTITUTIONS  # In production, this would fetch from a database
        self._build_action_graph()

    def _format_institutions(self, institutions: List[Dict]) -> str:
        """Format the institutions into a string for the prompt"""
        formatted_institutions = ""
        for i, institution in enumerate(institutions):
            formatted_institutions += f"[Institution {i+1}]\n"
            formatted_institutions += f"Name: {institution['name']}\n"
            formatted_institutions += f"Location: {institution['location']}\n"
            formatted_institutions += f"Areas of Expertise: {', '.join(institution['expertise'])}\n"
            formatted_institutions += f"Collaboration Opportunities: {', '.join(institution['collaboration_opportunities'])}\n"
            formatted_institutions += f"Contact: {institution['contact']}\n\n"
        return formatted_institutions

    def _find_collaborations(self, state: MessageState) -> Dict[str, Any]:
        """Find and recommend collaboration opportunities based on user query"""
        user_message = state["user_message"]
        
        # Format institutions for the prompt
        formatted_institutions = self._format_institutions(self.institutions)
        
        prompt = PromptTemplate.from_template(research_collaboration_prompt)
        input_prompt = prompt.invoke({
            "message": user_message.message, 
            "history": user_message.history,
            "institutions": formatted_institutions
        })
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        
        final_chain = self.llm | StrOutputParser()
        recommendation = final_chain.invoke(chunked_prompt)
        
        # Update message state
        state["status"] = StatusEnum.completed
        state["answer"] = recommendation
        return state

    def _build_action_graph(self):
        """Build the action graph for finding collaborations"""
        find_collaborations_node = node_from_function(self._find_collaborations, name="find_collaborations")
        
        # Create the workflow
        workflow = StateGraph(MessageState)
        workflow.add_node("find_collaborations", find_collaborations_node)
        
        # Add entry point
        workflow.set_entry_point("find_collaborations")
        
        self.action_graph = workflow

    def execute(self, msg_state: MessageState):
        """Execute the research collaboration workflow"""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result 