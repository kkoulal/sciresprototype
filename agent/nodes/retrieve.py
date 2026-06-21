import httpx
from qdrant_client import QdrantClient, models
from agent.state import AgentState
from online.config import QDRANT, EMBED

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

def _build_domain_filter(domain_hints: list[str]) -> models.Filter | None:
    """Build a Qdrant OR-filter across domain_hints using MatchText conditions."""
    if not domain_hints:
        return None
    conditions = [
        models.FieldCondition(
            key="domain_classification",
            match=models.MatchText(text=hint),
        )
        for hint in domain_hints[:4]  # cap at 4 to avoid over-constraining
    ]
    if len(conditions) == 1:
        return models.Filter(must=conditions)
    return models.Filter(should=conditions)

def retrieve_node(state: AgentState) -> AgentState:
    qdrant = QdrantClient(host=QDRANT["host"], port=QDRANT["port"])

    # Build a domain filter from the parsed scenario's domain_hints.
    # On retry rounds (round >= 1) we drop the filter to widen recall.
    scenario = state.get("parsed_scenario")
    domain_hints = list(scenario.domain_hints) if scenario else []

    for sq in state.get("sub_queries", []):
        round_num = state["retrieve_rounds"].get(sq.id, 0)
        if len(state["retrieved_docs"].get(sq.id, [])) > 0 or round_num >= 3:
            continue

        # Round 0: domain-scoped search for precision
        # Round 1+: no domain filter to broaden recall on retry
        qfilter = _build_domain_filter(domain_hints) if round_num == 0 else None

        try:
            r = httpx.post(f"{EMBED['url']}/embed_both", json={"inputs": [sq.query]}, timeout=30.0)
            r.raise_for_status()
            vectors = r.json()
            dense = vectors["dense"][0]
            sparse_dict = {item["index"]: item["value"] for item in vectors["sparse"][0]}
            sparse = models.SparseVector(
                indices=list(sparse_dict.keys()),
                values=list(sparse_dict.values()),
            )

            resp = qdrant.query_points(
                collection_name=QDRANT["collection"],
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=15),
                    models.Prefetch(query=sparse, using="sparse", limit=15),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=qfilter,
                with_payload=[
                    "doi", "type_etude", "population_echantillon",
                    "methodes_analyse_statistique", "resultats_principaux",
                    "valeurs_numeriques_cles", "limitations_mentionnees",
                    "domain_classification",
                ],
                limit=10,
            )

            # Keep only papers that actually report numeric values
            docs = [
                pt.payload for pt in resp.points
                if pt.payload and _has_numeric_values(pt.payload)
            ]

            state["retrieved_docs"][sq.id] = docs
            state["retrieve_rounds"][sq.id] = round_num + 1

        except Exception as e:
            state["errors"].append(f"Retrieval failed for {sq.id}: {e}")

    return state
