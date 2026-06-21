import json
import concurrent.futures
from agent.state import AgentState
from agent.models import ExtractedParameter
from agent.llm import call_llm
from agent.prompts import EXTRACT_PARAMS_PROMPT, render_prompt

_NOT_REPORTED = {
    "not explicitly reported in the provided text.",
    "not reported in the paper.",
    "no specific numerical values are reported in the provided text.",
    "not explicitly reported in the provided text",
    "not reported.",
    "",
}

def _has_numeric_values(doc: dict) -> bool:
    v = (doc.get("valeurs_numeriques_cles") or "").strip().lower()
    return v not in _NOT_REPORTED

def _process_doc(sq_query, doc):
    important_keys = [
        "doi", "type_etude", "population_echantillon", "methodes_analyse_statistique",
        "resultats_principaux", "valeurs_numeriques_cles", "limitations_mentionnees"
    ]
    context_data = {k: doc.get(k) for k in important_keys if doc.get(k)}
    context = json.dumps(context_data, indent=2)[:6000]

    prompt = render_prompt(
        EXTRACT_PARAMS_PROMPT,
        query=sq_query,
        title=doc.get("type_etude", "Unknown Study"),
        doi=doc.get("doi", "Unknown"),
        context=context,
    )
    
    # Parameter list scales with the number of retrieved papers; the 4000-token
    # default truncates the JSON array -> parse fails -> empty list. Give it room.
    result_list = call_llm([{"role": "user", "content": prompt}], extract_json=True,
                           max_tokens=8000)
    extracted = []
    if isinstance(result_list, list):
        for item in result_list:
            if isinstance(item, dict):
                if "paper_doi" not in item or item["paper_doi"] == "N/A":
                    item["paper_doi"] = doc.get("doi", "Unknown")
                try:
                    extracted.append(ExtractedParameter(**item))
                except Exception:
                    pass
    return extracted

def extract_params_node(state: AgentState) -> AgentState:
    extracted = []
    futures = []
    
    # Fire off all parameter extractions to the LLM concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for sq in state.get("sub_queries", []):
            docs = state["retrieved_docs"].get(sq.id, [])
            # Skip docs with no numeric values before paying LLM cost
            qdrant_docs = [d for d in docs if d.get("_source") != "neo4j_graph" and _has_numeric_values(d)][:4]
            graph_docs  = [d for d in docs if d.get("_source") == "neo4j_graph"  and _has_numeric_values(d)][:2]
            for doc in qdrant_docs + graph_docs:
                futures.append(executor.submit(_process_doc, sq.query, doc))
                
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res:
                    extracted.extend(res)
            except Exception:
                pass
                
    state["extracted_parameters"] = extracted
    return state
