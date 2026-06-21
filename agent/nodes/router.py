from agent.state import AgentState
from agent.llm import call_llm

def router_node(state: AgentState) -> AgentState:
    prompt = f"""Classify the user's input. Pick ONE intent.

- "qa": the user is asking a question — anything from "what is X" to "find papers on Y" to "what fields do you know" to "compare A and B". This is the DEFAULT for any question. Use this for meta-questions about the corpus too.
- "explore": the user explicitly asked to cluster, group, or visualize the entire corpus broadly (e.g. "cluster all papers by topic", "show me a map of the corpus").
- "simulate": the user explicitly described a POLICY SCENARIO they want to simulate as an agent-based or system-dynamics model. Trigger words: "simulate this policy", "build a simulation model for X intervention", "model the impact of Y". Do NOT use this for questions about research, methods, or what to study next — those are "qa".

User Input: {state['raw_scenario']}

Respond ONLY with a JSON object: {{"intent": "qa" | "explore" | "simulate"}}"""

    result = call_llm([{"role": "user", "content": prompt}], extract_json=True)
    state["intent"] = result.get("intent", "qa") if isinstance(result, dict) else "qa"
    return state
