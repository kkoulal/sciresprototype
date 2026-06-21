from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    router_node,
    qa_node,
    explore_node,
    parse_scenario_node,
    plan_retrieval_node,
    retrieve_node,
    graph_enrich_node,
    extract_params_node,
    validate_node,
    build_package_node
)

def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("router", router_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("explore", explore_node)
    
    workflow.add_node("parse", parse_scenario_node)
    workflow.add_node("plan", plan_retrieval_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("graph_enrich", graph_enrich_node)
    workflow.add_node("extract", extract_params_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("build", build_package_node)
    
    workflow.set_entry_point("router")
    
    # Dynamic routing
    def route_intent(state: AgentState):
        intent = state.get("intent", "qa")
        if intent == "simulate": return "parse"
        elif intent == "explore": return "explore"
        else: return "qa"
            
    workflow.add_conditional_edges(
        "router",
        route_intent,
        {
            "parse": "parse",
            "explore": "explore",
            "qa": "qa"
        }
    )
    
    workflow.add_edge("qa", END)
    workflow.add_edge("explore", END)
    
    # Simulation pipeline
    workflow.add_edge("parse", "plan")
    workflow.add_edge("plan", "retrieve")
    
    def check_retrieval(state: AgentState):
        for sq in state.get("sub_queries", []):
            round_num = state["retrieve_rounds"].get(sq.id, 0)
            docs = state["retrieved_docs"].get(sq.id, [])
            if len(docs) == 0 and round_num < 3:
                return "retrieve"
        return "graph_enrich"

    workflow.add_conditional_edges(
        "retrieve",
        check_retrieval,
        {"retrieve": "retrieve", "graph_enrich": "graph_enrich"}
    )

    workflow.add_edge("graph_enrich", "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", "build")
    workflow.add_edge("build", END)
    
    return workflow.compile()
