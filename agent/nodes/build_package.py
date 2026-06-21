from agent.state import AgentState
from agent.models import SimulationPackage
from agent.llm import call_llm
from agent.prompts import BUILD_PACKAGE_PROMPT, render_prompt

_RELIABILITY_RANK = {"high": 0, "medium": 1, "low": 2, "unknown": 3}

def _top_params(params_list, max_params: int = 20) -> list:
    """
    Sort extracted parameters by reliability (high > medium > low > unknown),
    then by study design quality (RCT > meta-analysis > observational > rest),
    and cap at max_params to keep the LLM prompt manageable.
    """
    _DESIGN_RANK = {"rct": 0, "meta-analysis": 1, "systematic review": 1,
                    "observational": 2, "experimental": 2, "modelling": 3, "modeling": 3}

    def _score(p):
        rel = _RELIABILITY_RANK.get(str(p.reliability).lower(), 3)
        design = next(
            (v for k, v in _DESIGN_RANK.items() if k in str(p.study_design).lower()), 4
        )
        # Prefer params with actual numeric values over string descriptions
        is_numeric = isinstance(p.value, (int, float))
        return (rel, design, 0 if is_numeric else 1)

    ranked = sorted(params_list, key=_score)
    return ranked[:max_params]

def build_package_node(state: AgentState) -> AgentState:
    scenario = state.get("parsed_scenario")
    if not scenario:
        return state

    params_list = state.get("extracted_parameters", [])
    top = _top_params(params_list, max_params=20)
    params_text = "\n".join([p.model_dump_json() for p in top])[:10000]

    prompt = render_prompt(
        BUILD_PACKAGE_PROMPT,
        scenario_title=scenario.scenario_title,
        intervention=scenario.intervention,
        notes=state.get("validation_notes", "None"),
        parameters=params_text,
    )
    
    # The full simulation package (params + calibration targets + model brief +
    # agent rules + uncertainty) easily exceeds the 4000-token default and gets
    # truncated mid-JSON -> json.loads fails -> empty result. Give it real room.
    result_dict = call_llm([{"role": "user", "content": prompt}], extract_json=True,
                           max_tokens=16000)
    if isinstance(result_dict, dict):
        try:
            state["simulation_package"] = SimulationPackage(**result_dict)
        except Exception as e:
            state["errors"].append(f"Failed to build package model: {e}")
            
    return state
