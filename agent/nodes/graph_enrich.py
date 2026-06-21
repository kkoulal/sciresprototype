from neo4j import GraphDatabase
from agent.state import AgentState
from online.config import NEO4J

def _neo4j():
    return GraphDatabase.driver(NEO4J["uri"], auth=(NEO4J["user"], NEO4J["password"]))

def graph_enrich_node(state: AgentState) -> AgentState:
    """
    Enrich retrieved_docs with Neo4j graph neighbors BEFORE parameter extraction.

    Two expansion strategies:
      1. Entity-neighbor: papers sharing high-frequency SimEntity nodes (USES edges)
         with the Qdrant-retrieved set — structural methodology peers.
      2. Future-continuation: papers that built on retrieved papers via
         FUTURE_REALIZED_IN — methodologically mature, downstream research.

    Per sub-query the final doc list is capped as:
      top-4 Qdrant (semantic relevance) + top-2 Neo4j (graph impact, by FWCI).
    """
    all_retrieved_dois: set[str] = set()
    for docs in state["retrieved_docs"].values():
        for doc in docs:
            if doc.get("doi"):
                all_retrieved_dois.add(doc["doi"])

    if not all_retrieved_dois:
        return state

    driver = _neo4j()
    dois_list = list(all_retrieved_dois)

    try:
        with driver.session() as session:
            # ── 1. Entity-neighbor expansion ──────────────────────────────────
            entity_records = list(session.run(
                """
                UNWIND $dois AS doi
                MATCH (p:SimPaper {doi: doi})-[:USES]->(e:SimEntity)
                WHERE e.frequency >= 3
                WITH e ORDER BY e.frequency DESC LIMIT 20
                MATCH (e)<-[:USES]-(sibling:SimPaper)
                WHERE NOT sibling.doi IN $dois
                  AND sibling.resultats_principaux IS NOT NULL
                RETURN DISTINCT sibling.doi AS doi,
                       coalesce(sibling.fwci, 0.0) AS fwci
                ORDER BY fwci DESC
                LIMIT 30
                """,
                dois=dois_list,
            ))

            # ── 2. FUTURE_REALIZED_IN expansion ───────────────────────────────
            future_records = list(session.run(
                """
                UNWIND $dois AS doi
                MATCH (p:SimPaper {doi: doi})-[:FUTURE_REALIZED_IN]->(cont:SimPaper)
                WHERE NOT cont.doi IN $dois
                  AND cont.resultats_principaux IS NOT NULL
                RETURN DISTINCT cont.doi AS doi,
                       coalesce(cont.fwci, 0.0) AS fwci
                ORDER BY fwci DESC
                LIMIT 15
                """,
                dois=dois_list,
            ))

            new_dois = list(
                {r["doi"] for r in entity_records + future_records} - all_retrieved_dois
            )
            if not new_dois:
                driver.close()
                return state

            # ── 3. Fetch full payload for new papers ──────────────────────────
            payload_records = list(session.run(
                """
                UNWIND $dois AS doi
                MATCH (p:SimPaper {doi: doi})
                RETURN p.doi                      AS doi,
                       p.type_etude               AS type_etude,
                       p.population_echantillon   AS population_echantillon,
                       p.methodes_analyse_statistique AS methodes_analyse_statistique,
                       p.resultats_principaux     AS resultats_principaux,
                       p.valeurs_numeriques_cles  AS valeurs_numeriques_cles,
                       p.limitations_mentionnees  AS limitations_mentionnees,
                       coalesce(p.fwci, 0.0)      AS fwci
                ORDER BY fwci DESC
                """,
                dois=new_dois,
            ))

    except Exception as e:
        state["errors"].append(f"Graph enrichment failed: {e}")
        driver.close()
        return state

    driver.close()

    # Build payload dicts that match the Qdrant format expected by extract_params
    graph_docs = []
    for r in payload_records:
        if not (r["resultats_principaux"] or r["valeurs_numeriques_cles"]):
            continue
        graph_docs.append({
            "doi":                         r["doi"],
            "type_etude":                  r["type_etude"] or "",
            "population_echantillon":      r["population_echantillon"] or "",
            "methodes_analyse_statistique": r["methodes_analyse_statistique"] or "",
            "resultats_principaux":        r["resultats_principaux"] or "",
            "valeurs_numeriques_cles":     r["valeurs_numeriques_cles"] or "",
            "limitations_mentionnees":     r["limitations_mentionnees"] or "",
            "_source": "neo4j_graph",
            "_fwci":   r["fwci"],
        })

    if not graph_docs:
        return state

    # ── 4. Distribute: append graph docs to each sub-query list ──────────────
    # extract_params_node will then take top-4 Qdrant + top-2 graph per sub-query.
    sq_ids = [sq.id for sq in state.get("sub_queries", [])]
    for i, doc in enumerate(graph_docs):
        sq_id = sq_ids[i % len(sq_ids)]
        state["retrieved_docs"][sq_id].append(doc)

    return state
