from agent.state import AgentState
from agent.llm import call_llm
from agent.prompts import VALIDATE_RECONCILE_PROMPT, render_prompt

def validate_node(state: AgentState) -> AgentState:
    params_list = state.get("extracted_parameters", [])
    if not params_list:
        state["validation_notes"] = "No parameters were extracted."
        return state

    params_text = "\n".join([p.model_dump_json() for p in params_list])
    # Truncate if insanely long
    params_text = params_text[:10000]

    prompt = render_prompt(VALIDATE_RECONCILE_PROMPT, parameters=params_text)
    
    notes = call_llm([{"role": "user", "content": prompt}], max_tokens=1000)
    state["validation_notes"] = str(notes)
    
    return state
