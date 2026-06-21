from agent.state import AgentState
from agent.models import PolicyScenario
from agent.llm import call_llm
from agent.prompts import PARSE_SCENARIO_PROMPT, render_prompt

def parse_scenario_node(state: AgentState) -> AgentState:
    raw = state["raw_scenario"]
    prompt = render_prompt(PARSE_SCENARIO_PROMPT, raw_scenario=raw)
    
    result_dict = call_llm([{"role": "user", "content": prompt}], extract_json=True)
    
    if result_dict:
        try:
            state["parsed_scenario"] = PolicyScenario(**result_dict)
        except Exception as e:
            state["errors"].append(f"Invalid schema for scenario: {e}")
    else:
        state["errors"].append("Failed to parse scenario")
        
    return state
