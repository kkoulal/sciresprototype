import json
from qdrant_client import QdrantClient
from agent.state import AgentState
from agent.llm import call_llm
from online.config import QDRANT
from collections import defaultdict

def explore_node(state: AgentState) -> AgentState:
    query = state["raw_scenario"]
    qdrant = QdrantClient(host=QDRANT["host"], port=QDRANT["port"])
    
    try:
        # Qdrant's payload index has 3 corrupted points that cause the OffsetZero panic.
        # Fix: Fetch IDs without payload, then retrieve in batches, skipping the broken ones.
        resp = qdrant.query_points(
            collection_name=QDRANT["collection"],
            query=None,
            limit=10000,
            with_payload=False,
            with_vectors=False
        )
        ids = [pt.id for pt in getattr(resp, "points", [])]
        
        docs = []
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            try:
                res = qdrant.retrieve(
                    collection_name=QDRANT["collection"],
                    ids=batch_ids,
                    with_payload=["doi", "domain_classification"],
                    with_vectors=False
                )
                for pt in res:
                    if pt.payload:
                        docs.append(pt.payload)
            except Exception:
                # Batch failed, fall back to 1-by-1 to skip the specific corrupted ID
                for single_id in batch_ids:
                    try:
                        res = qdrant.retrieve(
                            collection_name=QDRANT["collection"],
                            ids=[single_id],
                            with_payload=["doi", "domain_classification"],
                            with_vectors=False
                        )
                        if res and res[0].payload:
                            docs.append(res[0].payload)
                    except Exception:
                        pass # Skip the corrupted point
        
        # Build deterministic clusters by domain, normalizing the text
        clusters = defaultdict(list)
        for doc in docs:
            raw_domain = doc.get("domain_classification", "Unknown").strip()
            # Normalize: "Computer Science / AI" -> "Computer Science"
            domain = raw_domain.split('/')[0].split('-')[0].strip().title()
            if not domain or len(domain) < 2:
                domain = "Unknown"
            doi = doc.get("doi", "N/A")
            clusters[domain].append(doi)
            
        # Format the base report
        lines = [f"Found {len(docs)} papers across {len(clusters)} domains.\n"]
        for domain, dois in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
            lines.append(f"### {domain} ({len(dois)} papers)")
            lines.append(f"Representative DOIs: {', '.join(dois[:5])}")
            if len(dois) > 5:
                lines.append(f"...and {len(dois)-5} more.")
            lines.append("")
            
        base_report = "\n".join(lines)
        
        prompt = f"""Polish the following cluster report for readability and scientific clarity based on the user's request.
Keep it dense and data-focused. Keep all counts and DOIs grounded. Do not invent facts.

User Request: {query}

Report Data:
{base_report}
"""
        placeholder = state.get("ui_placeholder")
        def stream_updater(full_text):
            if placeholder:
                display_text = full_text
                if "<think>" in display_text and "</think>" not in display_text:
                    display_text = display_text.replace("<think>", "*(DeepSeek is analyzing clusters...)*\n\n```text\n") + "\n```"
                else:
                    display_text = display_text.replace("<think>", "*(Analysis complete)*\n<!--").replace("</think>", "-->\n")
                placeholder.markdown(display_text + "▌")
                
        final_report = call_llm(
            [{"role": "user", "content": prompt}], 
            temperature=0.1, 
            max_tokens=1500,
            stream_callback=stream_updater
        )
        
        if placeholder:
            placeholder.markdown(final_report)
            
        return {"intent": "explore", "final_answer": final_report}
        
    except Exception as e:
        state["errors"].append(f"Explore Failed: {e}")
        state["final_answer"] = f"Failed to explore corpus: {e}"
        
    return state
