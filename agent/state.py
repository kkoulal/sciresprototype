from typing import TypedDict, List, Dict, Any, Optional
from agent.models import PolicyScenario, SubQuery, ExtractedParameter, SimulationPackage

class AgentState(TypedDict):
    # Inputs
    raw_scenario: str # Can be scenario OR general question
    chat_history: List[Dict[str, str]]  # [{role, content}, ...] prior turns

    # Routing
    intent: str # "qa", "explore", "simulate"
    
    # Processed Data (Simulate)
    parsed_scenario: Optional[PolicyScenario]
    sub_queries: List[SubQuery]
    
    # Retrieval
    retrieved_docs: Dict[str, List[Dict[str, Any]]] # query_id -> list of payload dicts
    retrieve_rounds: Dict[str, int] # query_id -> retry count
    
    # Extraction & Synthesis
    extracted_parameters: List[ExtractedParameter]
    validation_notes: str
    
    # Output
    simulation_package: Optional[SimulationPackage]
    final_answer: str # Used for QA and Explore modes
    ui_placeholder: Optional[Any] # QA and Explore modes
    
    # Session memory — accumulated across turns, injected into QA prompt
    session_memory: Dict[str, Any]

    # Errors/Control
    errors: List[str]
