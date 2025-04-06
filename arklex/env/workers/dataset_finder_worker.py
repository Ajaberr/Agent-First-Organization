import logging
import os
from typing import Dict, Any, List

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

# Mock database of climate datasets - in a real implementation, this would be fetched from an actual database
CLIMATE_DATASETS = [
    {
        "name": "NASA Global Climate Change Data",
        "description": "Comprehensive climate data including temperature records, carbon dioxide levels, and sea level measurements",
        "url": "https://climate.nasa.gov/vital-signs/",
        "provider": "NASA",
        "topics": ["global warming", "temperature", "CO2", "sea level", "ice sheets"],
        "format": ["CSV", "JSON"]
    },
    {
        "name": "NOAA Climate Data Online",
        "description": "Weather and climate data including temperature, precipitation, and wind measurements",
        "url": "https://www.ncdc.noaa.gov/cdo-web/",
        "provider": "NOAA",
        "topics": ["weather", "precipitation", "temperature", "wind", "historical data"],
        "format": ["CSV", "PDF"]
    },
    {
        "name": "World Bank Climate Change Data",
        "description": "Data on climate change vulnerability, adaptation, and mitigation strategies",
        "url": "https://data.worldbank.org/topic/climate-change",
        "provider": "World Bank",
        "topics": ["adaptation", "mitigation", "vulnerability", "economic impact"],
        "format": ["CSV", "XLS", "API"]
    },
    {
        "name": "Coupled Model Intercomparison Project (CMIP6)",
        "description": "Multi-model simulations of future climate scenarios",
        "url": "https://esgf-node.llnl.gov/projects/cmip6/",
        "provider": "WCRP",
        "topics": ["climate models", "future projections", "scenarios", "simulations"],
        "format": ["NetCDF", "HDF5"]
    },
    {
        "name": "Berkeley Earth Surface Temperature Dataset",
        "description": "Comprehensive historical Earth surface temperature data since 1750",
        "url": "http://berkeleyearth.org/data/",
        "provider": "Berkeley Earth",
        "topics": ["temperature", "historical data", "land surface", "ocean temperature"],
        "format": ["CSV", "NetCDF"]
    }
]

dataset_finder_prompt = """You are a climate dataset finder specialized in helping researchers find the most appropriate datasets for their climate research.

User's input: {message}
Previous conversation: {history}

Based on the user's query, search through the following climate datasets and recommend the most relevant ones:

{datasets}

If the user's query is vague, ask follow-up questions to clarify their research needs.
Provide specific details about the recommended datasets, including:
1. Dataset name and provider
2. Brief description of what the dataset contains
3. What research questions it can help answer
4. How to access the dataset (URLs when available)
5. Available data formats

Prioritize datasets that best match the user's specific research interests and needs.
"""

@register_worker
class DatasetFinderWorker(BaseWorker):
    description = "Helps researchers find and access relevant climate research datasets based on their specific research questions, variables of interest, and technical requirements"

    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self.datasets = CLIMATE_DATASETS  # In production, this would fetch from a database
        self._build_action_graph()

    def _format_datasets(self, datasets: List[Dict]) -> str:
        """Format the datasets into a string for the prompt"""
        formatted_datasets = ""
        for i, dataset in enumerate(datasets):
            formatted_datasets += f"[Dataset {i+1}]\n"
            formatted_datasets += f"Name: {dataset['name']}\n"
            formatted_datasets += f"Provider: {dataset['provider']}\n"
            formatted_datasets += f"Description: {dataset['description']}\n"
            formatted_datasets += f"URL: {dataset['url']}\n"
            formatted_datasets += f"Topics: {', '.join(dataset['topics'])}\n"
            formatted_datasets += f"Formats: {', '.join(dataset['format'])}\n\n"
        return formatted_datasets

    def _find_datasets(self, state: MessageState) -> Dict[str, Any]:
        """Find and recommend datasets based on user query"""
        user_message = state["user_message"]
        
        # Format datasets for the prompt
        formatted_datasets = self._format_datasets(self.datasets)
        
        prompt = PromptTemplate.from_template(dataset_finder_prompt)
        input_prompt = prompt.invoke({
            "message": user_message.message, 
            "history": user_message.history,
            "datasets": formatted_datasets
        })
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        
        final_chain = self.llm | StrOutputParser()
        recommendation = final_chain.invoke(chunked_prompt)
        
        # Update message state
        state["status"] = StatusEnum.completed
        state["answer"] = recommendation
        return state

    def _build_action_graph(self):
        """Build the action graph for dataset finding"""
        find_datasets_node = node_from_function(self._find_datasets, name="find_datasets")
        
        # Create the workflow
        workflow = StateGraph(MessageState)
        workflow.add_node("find_datasets", find_datasets_node)
        
        # Add entry point
        workflow.set_entry_point("find_datasets")
        
        self.action_graph = workflow

    def execute(self, msg_state: MessageState):
        """Execute the dataset finder workflow"""
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result 