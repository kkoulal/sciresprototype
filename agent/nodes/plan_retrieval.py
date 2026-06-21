from agent.state import AgentState
from agent.models import SubQuery
from agent.llm import call_llm
from agent.prompts import PLAN_RETRIEVAL_PROMPT, render_prompt

def plan_retrieval_node(state: AgentState) -> AgentState:
    scenario = state.get("parsed_scenario")
    if not scenario:
        return state

    prompt = render_prompt(
        PLAN_RETRIEVAL_PROMPT,
        scenario_title=scenario.scenario_title,
        intervention=scenario.intervention,
        outcomes=", ".join(scenario.outcome_of_interest),
        population=scenario.target_population,
    )
    
    result_list = call_llm([{"role": "user", "content": prompt}], extract_json=True)
    
    sub_queries = []
    if isinstance(result_list, list):
        for item in result_list:
            if isinstance(item, dict):
                try:
                    sub_queries.append(SubQuery(**item))
                except Exception as e:
                    state["errors"].append(f"SubQuery parse error: {e}")
    
    state["sub_queries"] = sub_queries
    
    # Initialize retrieval state
    if "retrieved_docs" not in state:
        state["retrieved_docs"] = {}
    if "retrieve_rounds" not in state:
        state["retrieve_rounds"] = {}
        
    for sq in sub_queries:
        if sq.id not in state["retrieved_docs"]:
            state["retrieved_docs"][sq.id] = []
        if sq.id not in state["retrieve_rounds"]:
            state["retrieve_rounds"][sq.id] = 0
            
    return state
